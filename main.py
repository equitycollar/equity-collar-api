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
max_loss_ps = (putK  - data.entry_price) + net_premium
max_gain_ps = (callK - data.entry_price) + net_premium

# --- totals ---
max_loss = max_loss_ps * data.shares
max_gain = max_gain_ps * data.shares

# --- true breakeven (only one for a standard collar) ---
breakeven = data.entry_price - net_premium

# --- payoff curve (must flatten at tails) ---
lo = min(putK * 0.6, breakeven)     # padding on the left
hi = max(callK * 1.4, breakeven)    # padding on the right
prices = np.linspace(lo, hi, 121)
payoff = []
for sT in prices:
    intrinsic_put  = max(putK  - sT, 0.0)   # ADD (long put)
    intrinsic_call = max(sT    - callK, 0.0) # SUBTRACT (short call)
    pnl_per_sh = (sT - data.entry_price) + intrinsic_put - intrinsic_call + net_premium
    payoff.append(round(float(pnl_per_sh * data.shares), 2))

return {
    # ... existing fields ...
    "selected_put_strike": putK,
    "selected_call_strike": callK,
    "put_bid": put_bid, "put_ask": put_ask, "put_last": put_last,
    "call_bid": call_bid, "call_ask": call_ask, "call_last": call_last,
    "put_premium_paid": round(put_premium_paid, 4),
    "call_premium_received": round(call_premium_rcv, 4),
    "net_premium": round(net_premium, 4),
    "max_loss": round(max_loss, 2),
    "max_gain": round(max_gain, 2),
    "breakeven_estimate": round(breakeven, 4),
    "payoff_prices": [round(float(x), 2) for x in prices],
    "payoff_values": payoff,
}

    

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Calculation error: {e}")
