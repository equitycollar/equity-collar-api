# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import yfinance as yf
import numpy as np

app = FastAPI(title="Equity Collar API")

# Open CORS for MVP; later restrict to your Vercel domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # e.g., ["https://your-frontend.vercel.app"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Models ----------
class CalcRequest(BaseModel):
    ticker: str = Field(..., examples=["AAPL"])
    shares: int = Field(..., gt=0, examples=[100])
    entry_price: float = Field(..., gt=0)
    put_strike: float = Field(..., gt=0)
    call_strike: float = Field(..., gt=0)
    expiration: str = Field(..., description="YYYY-MM-DD")

# ---------- Helpers ----------
def _mid(bid, ask, last):
    """Best-effort mid price from bid/ask/last."""
    try:
        b = float(bid) if bid is not None else np.nan
        a = float(ask) if ask is not None else np.nan
        l = float(last) if last is not None else np.nan
    except Exception:
        b = a = l = np.nan
    if np.isfinite(b) and np.isfinite(a) and a > 0:
        return (b + a) / 2.0
    if np.isfinite(l):
        return l
    if np.isfinite(b):
        return b
    if np.isfinite(a):
        return a
    return 0.0

def _nearest_row(df, strike):
    if df is None or len(df) == 0:
        return None
    idx = (df["strike"] - strike).abs().idxmin()
    return df.loc[idx].to_dict()

# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/expirations/{ticker}")
def expirations(ticker: str):
    t = yf.Ticker(ticker.upper().strip())
    exps = list(t.options or [])
    if not exps:
        raise HTTPException(status_code=404, detail="No option expirations found")
    return exps

@app.post("/calculate")
def calculate(data: CalcRequest):
    tkr = data.ticker.upper().strip()
    t = yf.Ticker(tkr)

    # Validate expiration
    exps = list(t.options or [])
    if data.expiration not in exps:
        raise HTTPException(status_code=400, detail="Expiration not available for this ticker")

    # Spot (delayed)
    price = t.history(period="1d")["Close"]
    spot = float(price.iloc[-1]) if len(price) else None

    # Chains
    try:
        chain = t.option_chain(data.expiration)
        calls_df = chain.calls
        puts_df = chain.puts
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to load option chain: {e}")

    # Select nearest strikes
    put_row  = _nearest_row(puts_df,  data.put_strike)
    call_row = _nearest_row(calls_df, data.call_strike)
    if put_row is None or call_row is None:
        raise HTTPException(status_code=500, detail="Could not find strikes in option chain")

    putK  = float(put_row["strike"])
    callK = float(call_row["strike"])

    # Premiums (credit positive: receive call, pay put)
    put_premium_paid = _mid(put_row.get("bid"),  put_row.get("ask"),  put_row.get("lastPrice"))
    call_premium_rcv = _mid(call_row.get("bid"), call_row.get("ask"), call_row.get("lastPrice"))
    net_premium = float(call_premium_rcv) - float(put_premium_paid)  # credit=+, debit=-

    # Per-share constants at tails
    max_loss_ps = (putK  - data.entry_price) + net_premium
    max_gain_ps = (callK - data.entry_price) + net_premium

    max_loss = round(max_loss_ps * data.shares, 2)
    max_gain = round(max_gain_ps * data.shares, 2)

    # True breakeven (single point for standard collar)
    breakeven = data.entry_price - net_premium

    # Price grid
    lo = float(min(putK * 0.6, breakeven))
    hi = float(max(callK * 1.4, breakeven))
    prices = np.linspace(lo, hi, 121)

    # Raw payoff (per share then × shares)
    payoff = []
    for sT in prices:
        intrinsic_put  = max(putK - sT, 0.0)    # long put → ADD
        intrinsic_call = max(sT - callK, 0.0)   # short call → SUB
        pnl_per_sh = (sT - data.entry_price) + intrinsic_put - intrinsic_call + net_premium
        payoff.append(round(float(pnl_per_sh * data.shares), 2))

    # HARD GUARD: flatten tails so chart cannot slope past caps
    payoff = [
        max_loss if sT <= putK else
        max_gain if sT >= callK else
        val
        for sT, val in zip(prices, payoff)
    ]

    return {
        "ticker": tkr,
        "shares": data.shares,
        "entry_price": data.entry_price,

        "selected_put_strike": putK,
        "selected_call_strike": callK,

        "put_premium_paid": round(float(put_premium_paid), 4),
        "call_premium_received": round(float(call_premium_rcv), 4),
        "net_premium": round(float(net_premium), 4),

        "max_loss": max_loss,
        "max_gain": max_gain,
        "breakeven_estimate": round(float(breakeven), 4),
        "spot_price": round(float(spot), 6) if spot is not None else None,

        "expiration": data.expiration,
        "payoff_prices": [round(float(x), 2) for x in prices],
        "payoff_values": payoff,
    }
