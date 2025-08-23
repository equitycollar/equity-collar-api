# main.py — FastAPI backend for Equity Collar Calculator (delayed data via yfinance)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import yfinance as yf
import numpy as np
import pandas as pd
from typing import List, Optional

app = FastAPI(title="Equity Collar API", version="1.0.0")

# CORS: allow your WordPress/React frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://equity-collar-frontend.vercel.app"],   # tighten to your domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CollarInput(BaseModel):
    ticker: str = Field(..., examples=["AAPL"])
    shares: int = Field(..., gt=0, examples=[100])
    entry_price: float = Field(..., gt=0, examples=[160.0])
    put_strike: float = Field(..., gt=0, examples=[150.0])
    call_strike: float = Field(..., gt=0, examples=[175.0])
    expiration: str = Field(..., description="YYYY-MM-DD", examples=["2025-12-20"])

def _mid(bid, ask, last):
    """Return best available price: bid/ask mid, else last, else 0."""
    try:
        b = float(bid) if pd.notna(bid) else np.nan
        a = float(ask) if pd.notna(ask) else np.nan
        l = float(last) if pd.notna(last) else np.nan
    except Exception:
        b = a = l = np.nan
    if pd.notna(b) and pd.notna(a) and a > 0:
        return (b + a) / 2.0
    if pd.notna(l):
        return l
    if pd.notna(a):
        return a
    if pd.notna(b):
        return b
    return 0.0

def _row_by_strike(df: pd.DataFrame, strike: float) -> pd.Series:
    """Get exact strike if present, otherwise the nearest strike."""
    if "strike" not in df.columns or df.empty:
        raise ValueError("Option chain is empty or malformed.")
    exact = df.loc[df["strike"] == float(strike)]
    if not exact.empty:
        return exact.iloc[0]
    # nearest
    idx = (df["strike"] - float(strike)).abs().idxmin()
    return df.loc[idx]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/expirations/{ticker}")
def expirations(ticker: str) -> List[str]:
    try:
        t = yf.Ticker(ticker)
        exps = t.options or []
        return exps
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not fetch expirations: {e}")

@app.get("/quote/{ticker}")
def quote(ticker: str):
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="1d")
        if hist.empty:
            raise HTTPException(status_code=404, detail="No quote data.")
        last = float(hist["Close"].iloc[-1])
        return {"ticker": ticker.upper(), "last": last}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Quote error: {e}")

@app.post("/calculate")
putK  = float(put_row["strike"])
callK = float(call_row["strike"])

# --- premiums (credit positive) ---
put_bid = float(put_row.get("bid", np.nan));   put_ask = float(put_row.get("ask", np.nan));   put_last = float(put_row.get("lastPrice", np.nan))
call_bid = float(call_row.get("bid", np.nan)); call_ask = float(call_row.get("ask", np.nan)); call_last = float(call_row.get("lastPrice", np.nan))
put_premium_paid = _mid(put_bid, put_ask, put_last)          # long put → pay
call_premium_rcv = _mid(call_bid, call_ask, call_last)       # short call → receive
net_premium = float(call_premium_rcv) - float(put_premium_paid)

# --- constants at tails (per-share) ---
# strikes
putK  = float(put_row["strike"])
callK = float(call_row["strike"])

# premiums (credit positive)
put_premium_paid = _mid(put_row.get("bid"), put_row.get("ask"), put_row.get("lastPrice"))
call_premium_rcv = _mid(call_row.get("bid"), call_row.get("ask"), call_row.get("lastPrice"))
net_premium = float(call_premium_rcv) - float(put_premium_paid)

# per-share constants at tails
max_loss_ps = (putK  - data.entry_price) + net_premium
max_gain_ps = (callK - data.entry_price) + net_premium

max_loss = round(max_loss_ps * data.shares, 2)
max_gain = round(max_gain_ps * data.shares, 2)

breakeven = data.entry_price - net_premium  # subtract (credit shifts BE up)

# price grid
lo = min(putK * 0.6, breakeven)
hi = max(callK * 1.4, breakeven)
prices = np.linspace(lo, hi, 121)

# payoff curve (mathematically correct)
payoff = []
for sT in prices:
    intrinsic_put  = max(putK - sT, 0.0)     # long put → ADD
    intrinsic_call = max(sT - callK, 0.0)    # short call → SUB
    pnl_per_sh = (sT - data.entry_price) + intrinsic_put - intrinsic_call + net_premium
    payoff.append(round(float(pnl_per_sh * data.shares), 2))

# HARD GUARD: enforce flat tails visually (can’t slope past caps)
for i, sT in enumerate(prices):
    if sT <= putK:
        payoff[i] = max_loss
    elif sT >= callK:
        payoff[i] = max_gain

    

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Calculation error: {e}")
