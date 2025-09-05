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
