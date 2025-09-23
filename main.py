# main.py — FastAPI backend (v3.4: better spot, premium block, full expirations, strikes, cache+retry+BSM, $1 grid)
from fastapi import FastAPI, HTTPException, Header, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import os, time, threading, datetime as dt, math
from math import log, sqrt, exp

app = FastAPI()
API_VERSION = "collar-api v3.4 (better spot + premium block)"

# ------------------------------------------------------------------------------
# CORS: prod + localhost + ANY Vercel preview URL for this project
# ------------------------------------------------------------------------------
ALLOWED_ORIGINS = [
    "https://equity-collar-frontend.vercel.app",
    "http://localhost:5173",
]
PREVIEW_ORIGIN_REGEX = r"^https://equity-collar-frontend-[a-z0-9-]+-equitycollars-projects\.vercel\.app$"

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=PREVIEW_ORIGIN_REGEX,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# Models
# ------------------------------------------------------------------------------
class CalcRequest(BaseModel):
    ticker: str
    shares: int
    entry_price: float
    put_strike: float
    call_strike: float
    expiration: str

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

# ------------------------------------------------------------------------------
# Health
# ------------------------------------------------------------------------------
@app.get("/")
def root():
    return {"ok": True, "version": API_VERSION, "docs": "/docs", "redoc": "/redoc"}

@app.get("/health")
def health():
    return {"status": "ok", "version": API_VERSION}

@app.get("/debug")
def debug_get():
    return {"ok": True, "version": API_VERSION}

# ------------------------------------------------------------------------------
# Date helpers
# ------------------------------------------------------------------------------
def _us_pretty(date_iso: str) -> str:
    try:
        d = dt.date.fromisoformat(date_iso)
        return f"{d.month}/{d.day}/{d.year}"  # no leading zeros
    except Exception:
        return date_iso

def _next_fridays_years(years: int = 2) -> List[str]:
    out = []
    today = dt.date.today()
    days_ahead = (4 - today.weekday()) % 7
    first = today + dt.timedelta(days=days_ahead or 7)
    end = today + dt.timedelta(weeks=52*years)
    cur = first
    while cur <= end:
        out.append(cur.isoformat())
        cur += dt.timedelta(weeks=1)
    return out

# ------------------------------------------------------------------------------
# Expirations & Strikes
# ------------------------------------------------------------------------------
@app.get("/expirations/{ticker}")
def expirations_path(ticker: str):
    try:
        tkr = yf.Ticker(ticker, session=_yf_session())
        opts = list(tkr.options or [])
        if not opts:
            opts = _next_fridays_years(2)
        return {"symbol": ticker.upper(),
                "expirations": opts,
                "expirations_pretty": [_us_pretty(x) for x in opts]}
    except Exception:
        opts = _next_fridays_years(2)
        return {"symbol": ticker.upper(),
                "expirations": opts,
                "expirations_pretty": [_us_pretty(x) for x in opts]}

@app.get("/expirations")
def expirations_query(symbol: Optional[str] = None, ticker: Optional[str] = None):
    sym = (symbol or ticker or "").upper()
    if not sym:
        raise HTTPException(status_code=400, detail="symbol (or ticker) query param is required")
    try:
        tkr = yf.Ticker(sym, session=_yf_session())
        opts = list(tkr.options or [])
    except Exception:
        opts = []
    if not opts:
        opts = _next_fridays_years(2)
    return {"symbol": sym, "expirations": opts, "expirations_pretty": [_us_pretty(x) for x in opts]}

@app.get("/strikes")
def strikes(symbol: str = Query(...), expiration: str = Query(...)):
    """Return available strikes for symbol+expiration (puts, calls, union)."""
    tkr = yf.Ticker(symbol, session=_yf_session())
    try:
        opt = tkr.option_chain(expiration)
        calls, puts = opt.calls, opt.puts
    except Exception:
        calls, puts, _, _ = fetch_option_chain(symbol, expiration, spot_hint=None)
    put_list = sorted(set([float(x) for x in (puts["strike"].tolist() if not puts.empty else [])]))
    call_list = sorted(set([float(x) for x in (calls["strike"].tolist() if not calls.empty else [])]))
    all_list = sorted(set(put_list) | set(call_list))
    return {"symbol": symbol.upper(), "expiration": expiration,
            "puts": put_list, "calls": call_list, "all": all_list}

# ------------------------------------------------------------------------------
# Retry session for Yahoo
# ------------------------------------------------------------------------------
def _yf_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=3, backoff_factor=0.4,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"]
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s

# ------------------------------------------------------------------------------
# Cache
# ------------------------------------------------------------------------------
_CACHE_LOCK = threading.Lock()
_CHAIN_CACHE: Dict[Tuple[str, str], Dict[str, Any]] = {}

def _cache_key(ticker: str, expiration: str) -> Tuple[str, str]:
    return (ticker.upper(), expiration)

def _cache_put_chain(ticker: str, expiration: str, calls_df: pd.DataFrame, puts_df: pd.DataFrame, spot: Optional[float]):
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
    calls_df = pd.DataFrame(entry["calls"]) if entry.get("calls") else pd.DataFrame()
    puts_df  = pd.DataFrame(entry["puts"])  if entry.get("puts")  else pd.DataFrame()
    age_sec = int(time.time() - entry["ts"])
    return {"calls": calls_df, "puts": puts_df, "spot": entry.get("spot"), "age_sec": age_sec}

def _cache_get_nearest_chain(ticker: str, expiration: str):
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

# ------------------------------------------------------------------------------
# Math helpers
# ------------------------------------------------------------------------------
def _last_valid(series: pd.Series) -> Optional[float]:
    try:
        v = series.dropna()
        if not v.empty:
            return float(v.iloc[-1])
    except Exception:
        pass
    return None

def _safe_spot(tkr: yf.Ticker, fallback: Optional[float] = None) -> Optional[float]:
    """Best-effort spot: fast_info → 1m bars (5d) → 1d close → fallback."""
    # 1) fast_info hints
    try:
        fi = getattr(tkr, "fast_info", {}) or {}
        for key in ("last_price", "regular_market_price", "regularMarketPrice", "last_close", "previous_close", "previousClose"):
            v = fi.get(key)
            if v is not None and float(v) > 0:
                return float(v)
    except Exception:
        pass
    # 2) last 1-minute bar from recent days
    try:
        h = tkr.history(period="5d", interval="1m")
        if isinstance(h, pd.DataFrame) and not h.empty and "Close" in h:
            v = _last_valid(h["Close"])
            if v is not None:
                return v
    except Exception:
        pass
    # 3) last daily close
    try:
        h = tkr.history(period="1d")
        if isinstance(h, pd.DataFrame) and not h.empty and "Close" in h:
            v = _last_valid(h["Close"])
            if v is not None:
                return v
    except Exception:
        pass
    # 4) fallback to provided hint
    return fallback

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _bsm_call_put(S: float, K: float, r: float, q: float, sigma: float, T_years: float) -> Tuple[float, float]:
    if T_years <= 0: T_years = 1.0 / 365.0
    if sigma <= 0:   sigma = 1e-4
    d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T_years) / (sigma * sqrt(T_years))
    d2 = d1 - sigma * sqrt(T_years)
    Nd1, Nd2 = _norm_cdf(d1), _norm_cdf(d2)
    Nmd1, Nmd2 = _norm_cdf(-d1), _norm_cdf(-d2)
    disc_r = exp(-r * T_years); disc_q = exp(-q * T_years)
    call = S * disc_q * Nd1 - K * disc_r * Nd2
    put  = K * disc_r * Nmd2 - S * disc_q * Nmd1
    return float(call), float(put)

def _pick_row_exact(df: pd.DataFrame, strike: float) -> Optional[pd.Series]:
    if df is None or df.empty: return None
    exact = df[df["strike"] == strike]
    if not exact.empty: return exact.iloc[0]
    return None

def _pick_row_nearest(df: pd.DataFrame, strike: float) -> Optional[pd.Series]:
    if df is None or df.empty: return None
    try:
        exact = df[df["strike"] == strike]
        if not exact.empty: return exact.iloc[0]
        s = df["strike"].astype(float)
        idx = (s - float(strike)).abs().idxmin()
        return df.loc[idx]
    except Exception:
        return None

def _mid_from_row(row: pd.Series) -> float:
    try:
        bid = float(row.get("bid", float("nan"))); ask = float(row.get("ask", float("nan")))
        if bid == bid and ask == ask and ask > 0: return (bid + ask) / 2.0
    except Exception: pass
    try:
        lp = float(row.get("lastPrice")); if lp == lp and lp > 0: return lp
    except Exception: pass
    for key in ("bid", "ask"):
        try:
            v = float(row.get(key, float("nan"))); if v == v and v > 0: return v
        except Exception: pass
    return 0.01

def _inject_greeks(resp: dict, greeks: dict) -> None:
    """Guarantee greeks both nested and top-level."""
    if not isinstance(greeks, dict): greeks = {}
    resp["greeks"] = {**resp.get("greeks", {}), **greeks}
    for k in ("delta", "gamma", "vega", "theta", "rho"):
        if k in resp["greeks"]:
            resp[k] = resp["greeks"][k]

def _attach_premium_block(resp: dict) -> None:
    """Also expose a `premium` object for UIs that expect premium.greeks/anchorlock/signals"""
    resp["premium"] = {
        "greeks": resp.get("greeks", {}),
        "anchorlock": resp.get("anchorlock", {}),
        "signals": resp.get("signals", {}),
    }

# ------------------------------------------------------------------------------
# Chain fetcher: live → cache → nearest-cache → synthetic BSM (uses spot_hint)
# ------------------------------------------------------------------------------
def fetch_option_chain(ticker: str, expiration: str, spot_hint: Optional[float] = None):
    tkr = yf.Ticker(ticker, session=_yf_session())
    spot = _safe_spot(tkr, spot_hint)

    # Available expirations
    try:
        avail = list(tkr.options or [])
    except Exception:
        avail = []

    req_exp = expiration
    target_exp = None
    if avail:
        if req_exp in avail: target_exp = req_exp
        else:
            try:
                want = dt.date.fromisoformat(req_exp)
                def dist(e):
                    try: return abs((dt.date.fromisoformat(e) - want).days)
                    except Exception: return 10**9
                target_exp = min(avail, key=dist)
            except Exception:
                target_exp = avail[0]

    # (1) live
    if target_exp:
        try:
            opt = tkr.option_chain(target_exp)
            calls, puts = opt.calls, opt.puts
            live_spot = _safe_spot(tkr, spot_hint if spot is None else spot)
            _cache_put_chain(ticker, target_exp, calls, puts, live_spot)
            src = "yfinance" if target_exp == req_exp else f"yfinance-nearest:{target_exp}"
            return calls, puts, live_spot, src
        except Exception:
            pass

    # (2) exact cache
    cached = _cache_get_chain(ticker, req_exp)
    if cached and not cached["calls"].empty and not cached["puts"].empty:
        return cached["calls"], cached["puts"], cached["spot"] or spot, f"cache:{cached['age_sec']}s"

    # (3) nearest cache
    near = _cache_get_nearest_chain(ticker, req_exp)
    if near and not near["calls"].empty and not near["puts"].empty:
        return near["calls"], near["puts"], near["spot"] or spot, f"cache-nearest:{near['expiration_used']}:{near['age_sec']}s"

    # (4) synthetic BSM
    try:
        try:
            exp_dt = dt.date.fromisoformat(req_exp)
            days = max((exp_dt - dt.date.today()).days, 7)
        except Exception:
            days = 30
        T = days / 365.0
        q = 0.0
        try:
            fi = getattr(tkr, "fast_info", {}) or {}
            q = float(fi.get("dividend_yield") or 0.0) or 0.0
        except Exception:
            q = 0.0
        r = 0.045; sigma = 0.25
        S = float(spot) if spot is not None else (float(spot_hint) if spot_hint is not None else 100.0)

        def synth_df(strikes: List[float], is_call: bool):
            rows = []
            for K in strikes:
                c, p = _bsm_call_put(S, float(K), r, q, sigma, T)
                mid = c if is_call else p
                bid = max(mid * 0.98, 0.01); ask = max(mid * 1.02, bid + 0.01)
                rows.append({"strike": float(K), "bid": bid, "ask": ask, "lastPrice": mid})
            return pd.DataFrame(rows)

        lo = max(S * 0.7, 1); hi = S * 1.3
        grid = [round(x, 2) for x in np.linspace(lo, hi, 12)] + [50, 100, 150, 200, 250, 300]
        calls_df = synth_df(grid, True); puts_df = synth_df(grid, False)
        return calls_df, puts_df, S, "model:BSM(σ=0.25)"
    except Exception:
        raise HTTPException(status_code=503, detail="Unable to provide live or synthetic option data.")

# ------------------------------------------------------------------------------
# Engines
# ------------------------------------------------------------------------------
def _calc_from_chain(data: CalcRequest) -> Dict[str, Any]:
    tkr = data.ticker.upper()
    calls, puts, spot, source = fetch_option_chain(tkr, data.expiration, spot_hint=data.entry_price)

    # FORCE: entry price used = spot (if available), else requested entry
    entry_used = float(spot) if spot is not None else float(data.entry_price)

    # Use exact strikes if possible; otherwise nearest (but advise UI to call /strikes)
    put_row_exact  = _pick_row_exact(puts,  data.put_strike)
    call_row_exact = _pick_row_exact(calls, data.call_strike)
    put_row  = put_row_exact  if put_row_exact  is not None else _pick_row_nearest(puts,  data.put_strike)
    call_row = call_row_exact if call_row_exact is not None else _pick_row_nearest(calls, data.call_strike)
    if put_row is None or call_row is None:
        raise HTTPException(status_code=400, detail="No usable strikes (even from cache/synthetic). Use /strikes to list available strikes.")

    put_mid  = _mid_from_row(put_row)
    call_mid = _mid_from_row(call_row)
    net_premium = call_mid - put_mid

    # $1 price grid
    lo_ref = min(data.put_strike, entry_used, spot if spot is not None else entry_used)
    hi_ref = max(data.call_strike, entry_used, spot if spot is not None else entry_used)
    lo = int(max(1, math.floor(lo_ref * 0.75)))
    hi = int(math.ceil(hi_ref * 1.25))
    if hi - lo > 800: hi = lo + 800
    prices = np.arange(lo, hi + 1, 1, dtype=float)

    payoff = []
    for sT in prices:
        intrinsic_put = max(data.put_strike - sT, 0.0)
        intrinsic_call = max(sT - data.call_strike, 0.0)
        pnl_per_sh = (sT - entry_used) + intrinsic_put - intrinsic_call + net_premium
        payoff.append(round(pnl_per_sh * data.shares, 2))

    max_loss = round((data.put_strike - entry_used - net_premium) * data.shares, 2)
    max_gain = round((data.call_strike - entry_used - net_premium) * data.shares, 2)
    payoff_arr = np.array(payoff)
    payoff_arr[prices <= data.put_strike] = max_loss
    payoff_arr[prices >= data.call_strike] = max_gain

    # Available strikes (so UI can populate valid lists)
    put_strikes  = sorted(set([float(x) for x in (puts["strike"].tolist() if not puts.empty else [])]))
    call_strikes = sorted(set([float(x) for x in (calls["strike"].tolist() if not calls.empty else [])]))

    return {
        "ticker": tkr,
        "spot_price": float(entry_used),  # used as entry
        "entry_price_requested": float(data.entry_price),
        "entry_price_used": float(entry_used),
        "shares": data.shares,
        "expiration": data.expiration,
        "selected_put_strike": float(data.put_strike),
        "selected_call_strike": float(data.call_strike),
        "used_put_strike": float(put_row["strike"]),
        "used_call_strike": float(call_row["strike"]),
        "available_put_strikes": put_strikes,
        "available_call_strikes": call_strikes,
        "put_premium_paid": round(put_mid, 3),
        "call_premium_received": round(call_mid, 3),
        "net_premium": round(net_premium, 3),
        "max_loss": max_loss,
        "max_gain": max_gain,
        "breakeven_estimate": round(entry_used + net_premium, 3),
        "breakeven_low": float(data.put_strike),
        "breakeven_high": float(data.call_strike),
        "payoff_prices": [float(x) for x in prices],
        "payoff_values": [float(x) for x in payoff_arr.tolist()],
        "data_source": source,
    }

def premium_legacy(data: CalcRequest) -> Dict[str, Any]:
    base = _calc_from_chain(data)

    # Greeks (stubs; replace with real calc later)
    rng = np.random.default_rng(seed=(abs(hash((data.ticker, data.expiration))) % 2_147_483_647))
    g = {
        "delta": round(float(rng.uniform(-0.5, 0.5)), 3),
        "gamma": round(float(rng.uniform(0, 0.1)), 3),
        "vega":  round(float(rng.uniform(0, 1)), 3),
        "theta": round(float(rng.uniform(-1, 0)), 3),
        # "rho": round(float(rng.uniform(-0.2, 0.2)), 3),
    }
    _inject_greeks(base, g)

    # AnchorLock (deterministic placeholders)
    rsi = round(float(rng.uniform(20, 80)), 2)
    momentum = round(float(rng.uniform(-1, 1)), 3)
    earnings_strength = round(float(rng.uniform(0, 100)), 1)
    growth_factor = round(float(rng.uniform(0, 100)), 1)
    ma_200d = round(float(rng.uniform(150, 300)), 2)

    score = 50.0
    score += (rsi - 50.0) * 0.5
    score += momentum * 25.0
    score += (earnings_strength - 50.0) * 0.05
    score += (growth_factor - 50.0) * 0.05
    score = max(0.0, min(100.0, score))
    action = "WATCH"
    if score >= 70: action = "ROLL DOWN/CAP"
    if score <= 30: action = "ROLL UP/FLOOR"

    anchor = {
        "rsi": rsi,
        "momentum": momentum,
        "earnings_strength": earnings_strength,
        "growth_factor": growth_factor,
        "ma_200d": ma_200d,
        "score": round(float(score), 1),
        "action": action,
        "comments": "AnchorLock placeholder (legacy)",
    }
    drivers = {
        "rsi_bias": round((rsi - 50.0) * 0.5, 2),
        "momentum_bias": round(momentum * 25.0, 2),
        "earnings_bias": round((earnings_strength - 50.0) * 0.05, 2),
        "growth_bias": round((growth_factor - 50.0) * 0.05, 2),
    }

    base["anchorlock"] = anchor
    base["signals"] = {"score": anchor["score"], "action": anchor["action"], "drivers": drivers}
    _attach_premium_block(base)
    return base

def compute_payoff_v2(req: CalcV2Request) -> Dict[str, Any]:
    strikes = [l.strike for l in req.legs if l.strike is not None]
    lo = min([req.spot * 0.6] + [s * 0.75 for s in strikes]) if strikes else req.spot * 0.6
    hi = max([req.spot * 1.4] + [s * 1.25 for s in strikes]) if strikes else req.spot * 1.4
    lo_i = int(max(1, math.floor(lo))); hi_i = int(math.ceil(hi))
    if hi_i - lo_i > 800: hi_i = lo_i + 800
    prices = np.arange(lo_i, hi_i + 1, 1, dtype=float)

    upfront = -sum((l.premium or 0.0) * l.qty for l in req.legs)
    stock_qty = sum(l.qty for l in req.legs if l.type == "stock")

    points = []
    for px in prices:
        val = upfront + stock_qty * (px - req.spot)
        for l in req.legs:
            if l.type == "put" and l.strike is not None:  val += max(l.strike - px, 0.0) * l.qty
            if l.type == "call" and l.strike is not None: val += max(px - l.strike, 0.0) * l.qty
        points.append({"price": round(float(px), 2), "value": round(float(val), 2)})

    greeks = {"delta": 0.25, "gamma": 0.01, "vega": 0.12, "theta": -0.03, "rho": 0.05}  # stub
    return {"payoff": points, "greeks": greeks, "pnl_at_spot": 0.0, "notes": "v2 stub"}

def premium_v2(req: CalcV2Request) -> Dict[str, Any]:
    base = compute_payoff_v2(req)
    _inject_greeks(base, base.get("greeks", {}))  # ensure top-level greek aliases

    a = req.anchorlock or {}
    floor = float(a.get("floor", 0.80)); cap = float(a.get("cap", 1.10)); rebalance_trigger = float(a.get("rebalance_trigger", 0.05))
    momentum = 0.0; rsi = 50.0

    proximity_floor = max(0.0, (floor - 0.95) * 200.0)
    proximity_cap   = max(0.0, (1.05 - cap) * 200.0)
    score = 55.0 + momentum * 20.0 - (proximity_floor + proximity_cap)
    score = max(0.0, min(100.0, score))
    action = "WATCH"
    if score >= 70: action = "ROLL DOWN/CAP"
    if score <= 30: action = "ROLL UP/FLOOR"

    anchor = {
        "floor": floor, "cap": cap, "rebalance_trigger": rebalance_trigger,
        "rsi": rsi, "momentum": momentum, "score": round(float(score), 1),
        "action": action, "comments": "AnchorLock stub (v2)",
    }
    drivers = {
        "proximity_floor": round(proximity_floor, 2),
        "proximity_cap": round(proximity_cap, 2),
        "momentum": round(momentum * 20.0, 2),
    }

    base["anchorlock"] = anchor
    base["signals"] = {"score": anchor["score"], "action": anchor["action"], "drivers": drivers}
    _attach_premium_block(base)
    return base

# ------------------------------------------------------------------------------
# Unified endpoints
# ------------------------------------------------------------------------------
@app.post("/calculate")
def calculate_auto(payload: Dict[str, Any] = Body(...)):
    if {"ticker","shares","entry_price","put_strike","call_strike","expiration"} <= set(payload.keys()):
        return _calc_from_chain(CalcRequest(**payload))
    if {"symbol","spot","legs"} <= set(payload.keys()):
        return compute_payoff_v2(CalcV2Request(**payload))
    raise HTTPException(status_code=400, detail="Unrecognized payload shape for /calculate")

@app.post("/premium/calculate")
def premium_calculate_unified(
    payload: Dict[str, Any] = Body(...),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-KEY"),
    x_premium_key: Optional[str] = Header(default=None, alias="X-Premium-Key"),
):
    expected = os.environ.get("PREMIUM_API_KEY")
    provided = (x_api_key or x_premium_key or payload.get("api_key") or payload.get("premium_key"))
    if expected and provided != expected:
        raise HTTPException(status_code=403, detail="Forbidden")

    if {"ticker","shares","entry_price","put_strike","call_strike","expiration"} <= set(payload.keys()):
        clean = {k:v for k,v in payload.items() if k not in ("api_key","premium_key")}
        return premium_legacy(CalcRequest(**clean))
    if {"symbol","spot","legs"} <= set(payload.keys()):
        clean = {k:v for k,v in payload.items() if k not in ("api_key","premium_key")}
        return premium_v2(CalcV2Request(**clean))
    raise HTTPException(status_code=400, detail="Unrecognized payload shape for /premium/calculate")

# ------------------------------------------------------------------------------
# GET helper
# ------------------------------------------------------------------------------
@app.get("/calculate")
def calculate_get_help():
    return {
        "error": "Use POST /calculate with JSON.",
        "example_payload": {
            "ticker": "AAPL", "shares": 100, "entry_price": 190,
            "put_strike": 175, "call_strike": 205, "expiration": "2025-10-17"
        },
        "v2_example_payload": {
            "symbol": "AAPL", "spot": 190,
            "legs": [
                {"type":"stock","qty":100},
                {"type":"put","strike":175,"qty":-1,"premium":3.2},
                {"type":"call","strike":205,"qty":-1,"premium":2.8}
            ]
        }
    }
