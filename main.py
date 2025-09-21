# main.py — ONE FILE BACKEND (compat + health + safe expirations)
from fastapi import FastAPI, HTTPException, Header, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
import yfinance as yf
import os
from datetime import date, timedelta
import time, threading
import datetime as dt


app = FastAPI()
API_VERSION = "collar-api v2.0 (single-file, compat, safe-expirations)"

# ---------------- CORS ----------------
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    # Debug mode: allow any origin (no cookies). This removes CORS as a variable.
    allow_origins=["*"],
    allow_credentials=False,   # must be False when using "*"
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------- Models (legacy) ----------------
class CalcRequest(BaseModel):
    ticker: str
    shares: int
    entry_price: float
    put_strike: float
    call_strike: float
    expiration: str

# ---------------- Models (V2, legs-based) ----------------
class LegV2(BaseModel):
    type: str                      # 'stock' | 'call' | 'put'
    strike: Optional[float] = None
    qty: float
    premium: Optional[float] = None

class CalcV2Request(BaseModel):
    symbol: str
    spot: float
    legs: List[LegV2]
    expirations: Optional[List[str]] = None
    anchorlock: Optional[dict] = None

# ---------------- Premium security (accept both header names) ----------------
# --- drop-in, replaces require_premium_key and /premium/calculate ---

from typing import Optional, Dict, Any
from fastapi import Header, HTTPException, Body

def _expected_key() -> Optional[str]:
    return os.environ.get("PREMIUM_API_KEY") or None

def _check_key_or_allow_dev(provided: Optional[str]):
    expected = _expected_key()
    if not expected:
        return  # no gate when not configured (dev)
    if provided != expected:
        raise HTTPException(status_code=403, detail="Forbidden")

def _extract_header_key(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-KEY"),
    x_premium_key: Optional[str] = Header(default=None, alias="X-Premium-Key"),
) -> Optional[str]:
    return x_api_key or x_premium_key

# Unified premium route: accepts legacy or v2 payloads, key in header OR body
@app.post("/premium/calculate")
def premium_auto(
    payload: Dict[str, Any] = Body(...),
    header_key: Optional[str] = _extract_header_key,  # FastAPI injects headers here
):
    # Allow key in body too (no custom header => no CORS preflight)
    body_key = payload.get("api_key") or payload.get("premium_key")
    provided = header_key or body_key
    _check_key_or_allow_dev(provided)

    # legacy payload?
    if {"ticker","shares","entry_price","put_strike","call_strike","expiration"} <= set(payload.keys()):
        return premium_legacy(CalcRequest(**{k:v for k,v in payload.items() if k != "api_key" and k != "premium_key"}))
    # v2 payload?
    if {"symbol","spot","legs"} <= set(payload.keys()):
        return premium_v2(CalcV2Request(**{k:v for k,v in payload.items() if k != "api_key" and k != "premium_key"}))

    raise HTTPException(status_code=400, detail="Unrecognized payload shape for /premium/calculate")


# ---------------- Utils ----------------
# ---------- In-memory option-chain cache ----------
# Stores the last good chain per (ticker, expiration) so we can serve stale data.
# Reset on restart/redeploy (Render ephemeral memory).
_CACHE_LOCK = threading.Lock()
_CHAIN_CACHE: dict[tuple[str, str], dict] = {}  # {(ticker, exp): {"ts": epoch, "calls": [...], "puts": [...], "spot": float}}

def _cache_key(ticker: str, expiration: str) -> tuple[str, str]:
    return (ticker.upper(), expiration)

def _cache_put_chain(ticker: str, expiration: str, calls_df, puts_df, spot: float | None):
    data = {
        "ts": time.time(),
        "calls": calls_df[["strike", "bid", "ask", "lastPrice"]].to_dict("records") if "lastPrice" in calls_df else calls_df[["strike", "bid", "ask"]].to_dict("records"),
        "puts":  puts_df[["strike", "bid", "ask", "lastPrice"]].to_dict("records")  if "lastPrice" in puts_df  else puts_df[["strike", "bid", "ask"]].to_dict("records"),
        "spot": float(spot) if spot is not None else None,
    }
    with _CACHE_LOCK:
        _CHAIN_CACHE[_cache_key(ticker, expiration)] = data

def _cache_get_chain(ticker: str, expiration: str):
    with _CACHE_LOCK:
        entry = _CHAIN_CACHE.get(_cache_key(ticker, expiration))
    if not entry:
        return None
    # Rebuild DataFrames from records
    calls_df = pd.DataFrame(entry["calls"]) if entry.get("calls") else pd.DataFrame()
    puts_df  = pd.DataFrame(entry["puts"])  if entry.get("puts")  else pd.DataFrame()
    age_sec = int(time.time() - entry["ts"])
    return {"calls": calls_df, "puts": puts_df, "spot": entry.get("spot"), "age_sec": age_sec}

def _cache_get_nearest_chain(ticker: str, expiration: str):
    """If exact expiration not cached, use nearest cached expiration for this ticker."""
    want = None
    try:
        want = dt.date.fromisoformat(expiration)
    except Exception:
        pass
    best_key, best_delta = None, None
    with _CACHE_LOCK:
        for (tick, exp), entry in _CHAIN_CACHE.items():
            if tick != ticker.upper():
                continue
            try:
                exp_dt = dt.date.fromisoformat(exp)
            except Exception:
                continue
            delta = abs((exp_dt - want).days) if want else 10**9
            if best_delta is None or delta < best_delta:
                best_key, best_delta = (tick, exp), delta
        entry = _CHAIN_CACHE.get(best_key) if best_key else None
    if not entry:
        return None
    calls_df = pd.DataFrame(entry["calls"]) if entry.get("calls") else pd.DataFrame()
    puts_df  = pd.DataFrame(entry["puts"])  if entry.get("puts")  else pd.DataFrame()
    age_sec = int(time.time() - entry["ts"])
    return {"calls": calls_df, "puts": puts_df, "spot": entry.get("spot"), "age_sec": age_sec, "expiration_used": best_key[1]}


def _next_fridays(n=8):
    """Fallback dates: next N Fridays as ISO strings."""
    today = date.today()
    days_ahead = (4 - today.weekday()) % 7  # Friday = 4
    first_fri = today + timedelta(days=days_ahead or 7)
    return [(first_fri + timedelta(weeks=i)).isoformat() for i in range(n)]

# ---------- Utils ----------
def _safe_spot(tkr: yf.Ticker, fallback: float | None = None) -> float | None:
    try:
        v = getattr(tkr, "fast_info", {}).get("last_price")
        if v: return float(v)
    except Exception:
        pass
    try:
        return float(tkr.history(period="1d")["Close"].iloc[-1])
    except Exception:
        return fallback

def fetch_option_chain(ticker: str, expiration: str):
    """
    Best-effort chain fetch with built-in fallbacks:
    - Try requested expiration.
    - If not available, pick the nearest available expiration from Yahoo (live).
    - If Yahoo fails, use exact cached; if none, use nearest cached for this ticker.
    - Only raise 503 if we truly have nothing to serve.
    Returns (calls_df, puts_df, spot, source)
    source examples: 'yfinance', 'yfinance-nearest:2025-11-21', 'cache:123s', 'cache-nearest:2025-11-21:123s'
    """
    tkr = yf.Ticker(ticker)

    # ---------- figure out which expiration we can fetch live ----------
    req_exp = expiration
    avail = []
    try:
        avail = list(tkr.options or [])
    except Exception:
        avail = []

    target_exp = None
    if avail:
        if req_exp in avail:
            target_exp = req_exp
        else:
            # pick nearest ISO date to the requested one
            try:
                want = dt.date.fromisoformat(req_exp)
                def dist(e):
                    try:
                        return abs((dt.date.fromisoformat(e) - want).days)
                    except Exception:
                        return 10**9
                target_exp = min(avail, key=dist)
            except Exception:
                # if requested isn't a date, just take the soonest available
                target_exp = avail[0]

    # ---------- 1) try LIVE for target_exp (requested or nearest) ----------
    if target_exp:
        try:
            opt = tkr.option_chain(target_exp)
            calls, puts = opt.calls, opt.puts
            spot = _safe_spot(tkr, None)
            _cache_put_chain(ticker, target_exp, calls, puts, spot)
            src = "yfinance" if target_exp == req_exp else f"yfinance-nearest:{target_exp}"
            return calls, puts, spot, src
        except Exception:
            pass  # fall through

    # ---------- 2) exact cached ----------
    cached = _cache_get_chain(ticker, req_exp)
    if cached and not cached["calls"].empty and not cached["puts"].empty:
        return cached["calls"], cached["puts"], cached["spot"], f"cache:{cached['age_sec']}s"

    # ---------- 3) nearest cached ----------
    near = _cache_get_nearest_chain(ticker, req_exp)
    if near and not near["calls"].empty and not near["puts"].empty:
        return near["calls"], near["puts"], near["spot"], f"cache-nearest:{near['expiration_used']}:{near['age_sec']}s"

    # ---------- 4) truly nothing to serve ----------
    raise HTTPException(status_code=503, detail="No live option chain and no cached data yet. Try a different expiration or retry shortly.")

import math
from math import log, sqrt, exp

def _norm_cdf(x: float) -> float:
    # standard normal CDF via erf
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _bsm_call_put(S: float, K: float, r: float, q: float, sigma: float, T_years: float) -> tuple[float, float]:
    # Black–Scholes–Merton (continuous dividend yield q)
    if T_years <= 0:  # don't blow up on same-day
        T_years = 1.0 / 365.0
    if sigma <= 0:
        sigma = 0.0001
    d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T_years) / (sigma * sqrt(T_years))
    d2 = d1 - sigma * sqrt(T_years)
    Nd1, Nd2 = _norm_cdf(d1), _norm_cdf(d2)
    Nmd1, Nmd2 = _norm_cdf(-d1), _norm_cdf(-d2)
    disc_r = exp(-r * T_years)
    disc_q = exp(-q * T_years)
    call = S * disc_q * Nd1 - K * disc_r * Nd2
    put  = K * disc_r * Nmd2 - S * disc_q * Nmd1
    return float(call), float(put)



# ---------------- Health ----------------
@app.get("/")
def root():
    return {"ok": True, "version": API_VERSION, "docs": "/docs", "redoc": "/redoc"}

@app.get("/health")
def health():
    return {"status": "ok", "version": API_VERSION}

@app.get("/debug")
def debug_get():
    return {"ok": True, "version": API_VERSION}

# ---------------- Expirations (path & query styles, with safe fallback) ----------------
@app.get("/expirations/{ticker}")
def expirations_path(ticker: str):
    try:
        tkr = yf.Ticker(ticker)
        return tkr.options or _next_fridays()
    except Exception:
        return _next_fridays()

@app.get("/expirations")
def expirations_query(symbol: Optional[str] = None, ticker: Optional[str] = None):
    sym = (symbol or ticker or "").upper()
    if not sym:
        raise HTTPException(status_code=400, detail="symbol (or ticker) query param is required")
    try:
        tkr = yf.Ticker(sym)
        opts = tkr.options
    except Exception:
        opts = None
    return {"symbol": sym, "expirations": opts or _next_fridays()}

# ---------------- Legacy engine (your original behavior) ----------------
def calculate_legacy(data: CalcRequest) -> Dict[str, Any]:
    tkr = data.ticker.upper()
    calls, puts = fetch_option_chain(tkr, data.expiration)

    put_row = puts[puts["strike"] == data.put_strike]
    call_row = calls[calls["strike"] == data.call_strike]
    if put_row.empty or call_row.empty:
        raise HTTPException(status_code=400, detail="Strike not found in chain")

    put_mid = float((put_row["bid"] + put_row["ask"]) / 2)
    call_mid = float((call_row["bid"] + call_row["ask"]) / 2)

    net_premium = call_mid - put_mid
    max_loss = round((data.put_strike - data.entry_price - net_premium) * data.shares, 2)
    max_gain = round((data.call_strike - data.entry_price - net_premium) * data.shares, 2)

    spot = float(yf.Ticker(tkr).history(period="1d")["Close"].iloc[-1])

    prices = np.linspace(data.put_strike * 0.75, data.call_strike * 1.25, 100)
    payoff = []
    for sT in prices:
        intrinsic_put = max(data.put_strike - sT, 0.0)
        intrinsic_call = max(sT - data.call_strike, 0.0)
        pnl_per_sh = (sT - data.entry_price) + intrinsic_put - intrinsic_call + net_premium
        payoff.append(round(pnl_per_sh * data.shares, 2))

    payoff = np.array(payoff)
    payoff[prices <= data.put_strike] = max_loss
    payoff[prices >= data.call_strike] = max_gain

    return {
        "ticker": tkr,
        "spot_price": spot,
        "entry_price": data.entry_price,
        "shares": data.shares,
        "expiration": data.expiration,
        "selected_put_strike": data.put_strike,
        "selected_call_strike": data.call_strike,
        "put_premium_paid": round(put_mid, 3),
        "call_premium_received": round(call_mid, 3),
        "net_premium": round(net_premium, 3),
        "max_loss": max_loss,
        "max_gain": max_gain,
        "breakeven_estimate": round(data.entry_price + net_premium, 3),
        "breakeven_low": data.put_strike,
        "breakeven_high": data.call_strike,
        "payoff_prices": [round(x, 2) for x in prices],
        "payoff_values": payoff.tolist()
    }

def premium_legacy(data: CalcRequest) -> Dict[str, Any]:
    base = calculate_legacy(data)
    # Simple deterministic-ish stubs for Greeks (replace with real calc later)
    rng = np.random.default_rng(seed=(abs(hash((data.ticker, data.expiration))) % 2_147_483_647))
    base["delta"] = round(float(rng.uniform(-0.5, 0.5)), 3)
    base["gamma"] = round(float(rng.uniform(0, 0.1)), 3)
    base["vega"]  = round(float(rng.uniform(0, 1)), 3)
    base["theta"] = round(float(rng.uniform(-1, 0)), 3)
    # AnchorLock placeholders
    base["anchorlock"] = {
        "rsi": round(float(rng.uniform(20, 80)), 2),
        "momentum": round(float(rng.uniform(-1, 1)), 3),
        "earnings_strength": round(float(rng.uniform(0, 100)), 1),
        "growth_factor": round(float(rng.uniform(0, 100)), 1),
        "ma_200d": round(float(rng.uniform(150, 300)), 2),
    }
    return base

# ---------------- V2 engine (legs-based, used by the patched frontend) ----------------
def compute_payoff_v2(req: CalcV2Request) -> Dict[str, Any]:
    strikes = [l.strike for l in req.legs if l.strike is not None]
    lo = min([req.spot * 0.6] + [s * 0.75 for s in strikes]) if strikes else req.spot * 0.6
    hi = max([req.spot * 1.4] + [s * 1.25 for s in strikes]) if strikes else req.spot * 1.4
    prices = np.linspace(lo, hi, 60)

    upfront = -sum((l.premium or 0.0) * l.qty for l in req.legs)
    stock_qty = sum(l.qty for l in req.legs if l.type == "stock")

    points = []
    for px in prices:
        val = upfront + stock_qty * (px - req.spot)
        for l in req.legs:
            if l.type == "put" and l.strike is not None:
                val += max(l.strike - px, 0.0) * l.qty
            if l.type == "call" and l.strike is not None:
                val += max(px - l.strike, 0.0) * l.qty
        points.append({"price": round(float(px), 2), "value": round(float(val), 2)})

    greeks = {"delta": 0.25, "gamma": 0.01, "vega": 0.12, "theta": -0.03, "rho": 0.05}  # stub
    return {"payoff": points, "greeks": greeks, "pnl_at_spot": 0.0, "notes": "v2 stub"}

def premium_v2(req: CalcV2Request) -> Dict[str, Any]:
    base = compute_payoff_v2(req)
    a = req.anchorlock or {}
    anchor = {
        "floor": float(a.get("floor", 0.80)),
        "cap": float(a.get("cap", 1.10)),
        "rebalance_trigger": float(a.get("rebalance_trigger", 0.05)),
        "comments": "AnchorLock stub (v2)",
    }
    sig = {"score": 55.0, "action": "WATCH", "drivers": {
        "proximity_floor": 0.0, "time_pressure": 0.0, "coverage_gap": 0.0, "slope_risk": 0.0, "momentum": 0.0
    }}
    return {**base, "anchorlock": anchor, "signals": sig}

# ---------------- Unified endpoints (accept either payload shape) ----------------
@app.post("/calculate")
def calculate(data: CalcRequest):
    tkr = data.ticker.upper()

    # --- Try to get chain; don't crash if yfinance fails ---
    calls, puts = fetch_option_chain(tkr, data.expiration)

    # If we have a chain, validate strikes & compute mids
    if calls is not None and puts is not None:
        put_row = puts[puts["strike"] == data.put_strike]
        call_row = calls[calls["strike"] == data.call_strike]

        if put_row.empty or call_row.empty:
            # Return 400 with helpful context (not 500)
            try:
                avail_puts = sorted(puts["strike"].astype(float).tolist())
                avail_calls = sorted(calls["strike"].astype(float).tolist())
            except Exception:
                avail_puts, avail_calls = [], []
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Strike not found in chain",
                    "requested": {"put": data.put_strike, "call": data.call_strike},
                    "available_example": {
                        "puts_min_max": [avail_puts[:1], avail_puts[-1:]],
                        "calls_min_max": [avail_calls[:1], avail_calls[-1:]],
                    },
                },
            )

        put_mid = float((put_row["bid"] + put_row["ask"]) / 2)
        call_mid = float((call_row["bid"] + call_row["ask"]) / 2)
        source = "yfinance"
    else:
        # Fallback: no live chain → assume zero premiums
        put_mid = 0.0
        call_mid = 0.0
        source = "fallback:no_option_chain"

    net_premium = call_mid - put_mid

    # --- Spot price (robust) ---
    spot = None
    try:
        yt = yf.Ticker(tkr)
        # fast_info is quick; if missing, fall back to history
        spot = float(getattr(yt, "fast_info", {}).get("last_price"))  # may be None
    except Exception:
        spot = None
    if not spot:
        try:
            spot = float(yf.Ticker(tkr).history(period="1d")["Close"].iloc[-1])
        except Exception:
            # Last resort: use entry price to avoid crashing response
            spot = float(data.entry_price)

    # --- Payoff curve (with flat tails) ---
    import numpy as _np
    prices = _np.linspace(data.put_strike * 0.75, data.call_strike * 1.25, 100)
    payoff = []
    for sT in prices:
        intrinsic_put = max(data.put_strike - sT, 0.0)
        intrinsic_call = max(sT - data.call_strike, 0.0)
        pnl_per_sh = (sT - data.entry_price) + intrinsic_put - intrinsic_call + net_premium
        payoff.append(round(pnl_per_sh * data.shares, 2))

    payoff = _np.array(payoff)
    payoff[prices <= data.put_strike] = round((data.put_strike - data.entry_price - net_premium) * data.shares, 2)
    payoff[prices >= data.call_strike] = round((data.call_strike - data.entry_price - net_premium) * data.shares, 2)

    return {
        "ticker": tkr,
        "spot_price": float(spot),
        "entry_price": data.entry_price,
        "shares": data.shares,
        "expiration": data.expiration,
        "selected_put_strike": data.put_strike,
        "selected_call_strike": data.call_strike,
        "put_premium_paid": round(put_mid, 3),
        "call_premium_received": round(call_mid, 3),
        "net_premium": round(net_premium, 3),
        "max_loss": round((data.put_strike - data.entry_price - net_premium) * data.shares, 2),
        "max_gain": round((data.call_strike - data.entry_price - net_premium) * data.shares, 2),
        "breakeven_estimate": round(data.entry_price + net_premium, 3),
        "breakeven_low": data.put_strike,
        "breakeven_high": data.call_strike,
        "payoff_prices": [round(x, 2) for x in prices],
        "payoff_values": payoff.tolist(),
        "data_source": source,
    }


# --- unified premium route: accepts key via header OR body; no Depends needed ---
from typing import Optional, Dict, Any
from fastapi import Header, HTTPException, Body

@app.post("/premium/calculate")
def premium_calculate_unified(
    payload: Dict[str, Any] = Body(...),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-KEY"),
    x_premium_key: Optional[str] = Header(default=None, alias="X-Premium-Key"),
):
    # Accept key from header or body (body fields: api_key or premium_key)
    provided = (x_api_key or x_premium_key or
                payload.get("api_key") or payload.get("premium_key"))
    expected = os.environ.get("PREMIUM_API_KEY")

    # Allow if no key configured (dev). Enforce in prod.
    if expected and provided != expected:
        raise HTTPException(status_code=403, detail="Forbidden")

    # Route to legacy or v2 calculators
    if {"ticker","shares","entry_price","put_strike","call_strike","expiration"} <= set(payload.keys()):
        # Remove any key fields before passing to the model
        clean = {k:v for k,v in payload.items() if k not in ("api_key","premium_key")}
        return premium_legacy(CalcRequest(**clean))

    if {"symbol","spot","legs"} <= set(payload.keys()):
        clean = {k:v for k,v in payload.items() if k not in ("api_key","premium_key")}
        return premium_v2(CalcV2Request(**clean))

    raise HTTPException(status_code=400, detail="Unrecognized payload shape for /premium/calculate")


# ---------------- Legacy POST /debug preserved ----------------
@app.post("/debug")
def debug_legacy(data: CalcRequest):
    result = calculate_legacy(data)
    values = result["payoff_values"]
    return {
        "first5": values[:5],
        "last5": values[-5:],
        "max_loss": result["max_loss"],
        "max_gain": result["max_gain"]
    }
