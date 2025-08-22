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
def calculate(data: CollarInput):
    try:
        t = yf.Ticker(data.ticker)
        # validate expiration exists
        if data.expiration not in (t.options or []):
            raise HTTPException(status_code=400, detail="Expiration not available for this ticker.")

        chain = t.option_chain(data.expiration)
        puts = chain.puts
        calls = chain.calls
        if puts is None or calls is None or puts.empty or calls.empty:
            raise HTTPException(status_code=404, detail="Option chain unavailable.")

        put_row = _row_by_strike(puts, data.put_strike)
        call_row = _row_by_strike(calls, data.call_strike)

        # Use bid/ask mid as premium proxy; also return raw bid/ask for transparency
        put_ask = float(put_row.get("ask", np.nan))
        put_bid = float(put_row.get("bid", np.nan))
        put_last = float(put_row.get("lastPrice", np.nan))
        call_ask = float(call_row.get("ask", np.nan))
        call_bid = float(call_row.get("bid", np.nan))
        call_last = float(call_row.get("lastPrice", np.nan))

        put_premium_paid = _mid(put_bid, put_ask, put_last)  # long put -> pay
        call_premium_rcv = _mid(call_bid, call_ask, call_last)  # short call -> receive

        net_premium = call_premium_rcv - put_premium_paid

        # Max loss (S_T <= K_put): (K_put - entry + netPrem) * shares
        max_loss = (float(put_row["strike"]) - data.entry_price + net_premium) * data.shares
        # Max gain (S_T >= K_call): (K_call - entry + netPrem) * shares
        max_gain = (float(call_row["strike"]) - data.entry_price + net_premium) * data.shares

        # Breakeven logic (simple collar approximation):
        # If debit, breakeven shifts down; if credit, shifts up. Clip to strikes for readability.
        be = data.entry_price + net_premium
        breakeven_low = min(be, float(put_row["strike"]))
        breakeven_high = max(be, float(call_row["strike"]))

        # Provide payoff curve points (for charts)
        lo = min(float(put_row["strike"]) - 0.25 * abs(float(put_row["strike"])), be)  # some padding
        hi = max(float(call_row["strike"]) + 0.25 * abs(float(call_row["strike"])), be)
        prices = np.linspace(lo, hi, 101)
        payoff = []
        for sT in prices:
            intrinsic_put = max(float(put_row["strike"]) - sT, 0.0)
            intrinsic_call = max(sT - float(call_row["strike"]), 0.0)
            pnl = (sT - data.entry_price - intrinsic_put - 0.0 + 0.0 - intrinsic_call + net_premium) * data.shares
            # Note: signs: long stock (+), long put (- premium, + intrinsic), short call (+ premium, - intrinsic)
            # The above expression collapses premiums into net_premium and subtracts the short call intrinsic.
            payoff.append(round(float(pnl), 2))

        # Spot (delayed)
        hist = t.history(period="1d")
        spot = float(hist["Close"].iloc[-1]) if not hist.empty else None

        return {
            "ticker": data.ticker.upper(),
            "spot_price": spot,
            "entry_price": data.entry_price,
            "shares": data.shares,
            "expiration": data.expiration,
            "selected_put_strike": float(put_row["strike"]),
            "selected_call_strike": float(call_row["strike"]),
            "put_bid": put_bid, "put_ask": put_ask, "put_last": put_last,
            "call_bid": call_bid, "call_ask": call_ask, "call_last": call_last,
            "put_premium_paid": round(put_premium_paid, 4),
            "call_premium_received": round(call_premium_rcv, 4),
            "net_premium": round(float(net_premium), 4),
            "max_loss": round(float(max_loss), 2),
            "max_gain": round(float(max_gain), 2),
            "breakeven_estimate": round(float(be), 4),
            "breakeven_low": round(float(breakeven_low), 4),
            "breakeven_high": round(float(breakeven_high), 4),
            "payoff_prices": [round(float(x), 2) for x in prices],
            "payoff_values": payoff,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Calculation error: {e}")
