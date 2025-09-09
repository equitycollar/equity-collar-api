from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
import numpy as np
import pandas as pd
import yfinance as yf
import os

app = FastAPI()
API_VERSION = "collar-api v1.8 (premium + debug)"

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://equity-collar-frontend.vercel.app",
        "http://localhost:5173",   # vite dev
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-KEY"],
    allow_headers=["Content-Type", "X-API-KEY", "X-Premium-Key"],
)


# ---------- Models ----------
class CalcRequest(BaseModel):
    ticker: str
    shares: int
    entry_price: float
    put_strike: float
    call_strike: float
    expiration: str

# ---------- Security for Premium ----------
def require_premium_key(x_api_key: str | None = Header(default=None)):
    expected = os.environ.get("PREMIUM_API_KEY", "")
    if not expected or x_api_key != expected:
        raise HTTPException(status_code=403, detail="Forbidden")

# ---------- Utils ----------
def fetch_option_chain(ticker, expiration):
    tkr = yf.Ticker(ticker)
    try:
        opt = tkr.option_chain(expiration)
        return opt.calls, opt.puts
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Option chain fetch failed: {e}")

# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {"status": "ok", "version": API_VERSION}

@app.get("/expirations/{ticker}")
def expirations(ticker: str):
    try:
        tkr = yf.Ticker(ticker)
        return tkr.options
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Expirations fetch failed: {e}")

@app.post("/calculate")
def calculate(data: CalcRequest):
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

    # Payoff curve
    prices = np.linspace(data.put_strike * 0.75, data.call_strike * 1.25, 100)
    payoff = []
    for sT in prices:
        intrinsic_put = max(data.put_strike - sT, 0.0)
        intrinsic_call = max(sT - data.call_strike, 0.0)
        pnl_per_sh = (sT - data.entry_price) + intrinsic_put - intrinsic_call + net_premium
        payoff.append(round(pnl_per_sh * data.shares, 2))

    # enforce flat tails
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

@app.post("/premium/calculate")
def premium_calculate(data: CalcRequest, _=Depends(require_premium_key)):
    base = calculate(data)

    # Greeks (simplified stubs, replace with real calc later)
    base["delta"] = round(np.random.uniform(-0.5, 0.5), 3)
    base["gamma"] = round(np.random.uniform(0, 0.1), 3)
    base["vega"]  = round(np.random.uniform(0, 1), 3)
    base["theta"] = round(np.random.uniform(-1, 0), 3)

    # AnchorLock proprietary indicators (placeholders)
    base["anchorlock"] = {
        "rsi": round(np.random.uniform(20, 80), 2),
        "momentum": round(np.random.uniform(-1, 1), 3),
        "earnings_strength": round(np.random.uniform(0, 100), 1),
        "growth_factor": round(np.random.uniform(0, 100), 1),
        "ma_200d": round(np.random.uniform(150, 300), 2),
    }
    return base

@app.post("/debug")
def debug(data: CalcRequest):
    result = calculate(data)
    values = result["payoff_values"]
    return {
        "first5": values[:5],
        "last5": values[-5:],
        "max_loss": result["max_loss"],
        "max_gain": result["max_gain"]
    }
    # =======================
# COMPAT LAYER FOR V2 API
# =======================

# 1) GET / (root) and GET /debug for simple health checks
@app.get("/")
def root():
    return {"ok": True, "service": API_VERSION, "docs": "/docs", "redoc": "/redoc"}

@app.get("/debug")
def debug_get():
    return {"ok": True, "service": API_VERSION}

# 2) Query-style expirations wrapper: /expirations?symbol=XYZ → { expirations: [...] }
@app.get("/expirations")
def expirations_query(symbol: str | None = None, ticker: str | None = None):
    sym = (symbol or ticker or "").upper()
    if not sym:
        raise HTTPException(status_code=400, detail="symbol (or ticker) query param is required")
    try:
        tkr = yf.Ticker(sym)
        return {"symbol": sym, "expirations": tkr.options or []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Expirations fetch failed: {e}")

# 3) Accept BOTH premium header names (X-API-KEY and X-Premium-Key)
def require_premium_key_v2(
    x_api_key: str | None = Header(default=None, alias="X-API-KEY"),
    x_premium_key: str | None = Header(default=None, alias="X-Premium-Key"),
):
    expected = os.environ.get("PREMIUM_API_KEY", "")
    provided = x_api_key or x_premium_key
    if not expected or provided != expected:
        raise HTTPException(status_code=403, detail="Forbidden")

# 4) V2 models to match the frontend bundle
from typing import Optional, List
from pydantic import BaseModel

class LegV2(BaseModel):
    type: str                     # 'stock' | 'call' | 'put'
    strike: Optional[float] = None
    qty: float
    premium: Optional[float] = None  # per share/contract, credit(+)/debit(-)

class CalcV2Request(BaseModel):
    symbol: str
    spot: float
    legs: List[LegV2]
    expirations: Optional[List[str]] = None
    anchorlock: Optional[dict] = None

# 5) V2 payoff engine (simple intrinsic + stock, uses provided premiums)
def _compute_payoff_v2(req: CalcV2Request):
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
    return points

# 6) V2 endpoints that match the frontend contract
@app.post("/calculate_v2")
def calculate_v2(req: CalcV2Request):
    pts = _compute_payoff_v2(req)
    greeks = {"delta": 0.25, "gamma": 0.01, "vega": 0.12, "theta": -0.03, "rho": 0.05}  # stub
    return {"payoff": pts, "greeks": greeks, "pnl_at_spot": 0.0, "notes": "v2 stub"}

@app.post("/premium/calculate_v2")
def premium_calculate_v2(req: CalcV2Request, _=Depends(require_premium_key_v2)):
    base = calculate_v2(req)
    a = req.anchorlock or {}
    anchor = {
        "floor": float(a.get("floor", 0.80)),
        "cap": float(a.get("cap", 1.10)),
        "rebalance_trigger": float(a.get("rebalance_trigger", 0.05)),
        "comments": "AnchorLock stub (v2)",
    }
    # simple placeholder roll signal
    sig = {"score": 55.0, "action": "WATCH", "drivers": {
        "proximity_floor": 0.0, "time_pressure": 0.0, "coverage_gap": 0.0, "slope_risk": 0.0, "momentum": 0.0
    }}
    return {**base, "anchorlock": anchor, "signals": sig}

