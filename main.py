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

app = FastAPI()
API_VERSION = "collar-api v2.0 (single-file, compat, safe-expirations)"

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://equity-collar-frontend.vercel.app",
        "http://localhost:5173",
        # add custom/preview domains here if needed:
        # "https://your-domain.com",
        # "https://<project>-<hash>-vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["Content-Type", "X-API-KEY", "X-Premium-Key"],
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
def require_premium_key(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-KEY"),
    x_premium_key: Optional[str] = Header(default=None, alias="X-Premium-Key"),
):
    expected = os.environ.get("PREMIUM_API_KEY")
    if not expected:
        # If no key configured, allow (handy for dev). Change this if you want strict prod gating.
        return
    provided = x_api_key or x_premium_key
    if provided != expected:
        raise HTTPException(status_code=403, detail="Forbidden")

# ---------------- Utils ----------------
def _next_fridays(n=8):
    """Fallback dates: next N Fridays as ISO strings."""
    today = date.today()
    days_ahead = (4 - today.weekday()) % 7  # Friday = 4
    first_fri = today + timedelta(days=days_ahead or 7)
    return [(first_fri + timedelta(weeks=i)).isoformat() for i in range(n)]

def fetch_option_chain(ticker: str, expiration: str):
    tkr = yf.Ticker(ticker)
    try:
        opt = tkr.option_chain(expiration)
        return opt.calls, opt.puts
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Option chain fetch failed: {e}")

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
def calculate_auto(payload: Dict[str, Any] = Body(...)):
    # Legacy payload?
    if {"ticker","shares","entry_price","put_strike","call_strike","expiration"} <= set(payload.keys()):
        return calculate_legacy(CalcRequest(**payload))
    # V2 payload?
    if {"symbol","spot","legs"} <= set(payload.keys()):
        return compute_payoff_v2(CalcV2Request(**payload))
    raise HTTPException(status_code=400, detail="Unrecognized payload shape for /calculate")

@app.post("/premium/calculate")
def premium_auto(payload: Dict[str, Any] = Body(...), _=Depends(require_premium_key)):
    if {"ticker","shares","entry_price","put_strike","call_strike","expiration"} <= set(payload.keys()):
        return premium_legacy(CalcRequest(**payload))
    if {"symbol","spot","legs"} <= set(payload.keys()):
        return premium_v2(CalcV2Request(**payload))
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
