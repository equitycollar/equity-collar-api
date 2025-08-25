# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import yfinance as yf
import numpy as np

API_VERSION = "collar-api v1.5 (correct-put-sign + flat-tail clamp)"

app = FastAPI(title="Equity Collar API")

# Open CORS for MVP; tighten to your Vercel domain later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # e.g., ["https://your-frontend.vercel.app"]
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
    if np.isfinite(l): return l
    if np.isfinite(b): return b
    if np.isfinite(a): return a
    return 0.0

def _nearest_row(df, strike):
    if df is None or len(df) == 0:
        return None
    idx = (df["strike"] - strike).abs().idxmin()
    return df.loc[idx].to_dict()

# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {"status": "ok", "version": API_VERSION}

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

    # validate expiration
    exps = list(t.options or [])
    if data.expiration not in exps:
        raise HTTPException(status_code=400, detail="Expiration not available for this ticker")

    # spot (delayed)
    price = t.history(period="1d")["Close"]
    spot = float(price.iloc[-1]) if len(price) else None

    # chains
    try:
        chain = t.option_chain(data.expiration)
        calls_df = chain.calls
        puts_df  = chain.puts
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to load option chain: {e}")

    # closest strikes
    put_row  = _nearest_row(puts_df,  data.put_strike)
    call_row = _nearest_row(calls_df, data.call_strike)
    if put_row is None or call_row is None:
        raise HTTPException(status_code=500, detail="Could not find strikes in option chain")

    putK  = float(put_row["strike"])
    callK = float(call_row["strike"])

    # premiums (credit positive)
    put_premium_paid = _mid(put_row.get("bid"),  put_row.get("ask"),  put_row.get("lastPrice"))
    call_premium_rcv = _mid(call_row.get("bid"), call_row.get("ask"), call_row.get("lastPrice"))
    net_premium = float(call_premium_rcv) - float(put_premium_paid)  # credit=+, debit=-

    # per-share constants at tails
    max_loss_ps = (putK  - data.entry_price) + net_premium
    max_gain_ps = (callK - data.entry_price) + net_premium

    max_loss = round(max_loss_ps * data.shares, 2)
    max_gain = round(max_gain_ps * data.shares, 2)

    # correct breakeven: entry - net_premium
    breakeven = data.entry_price - net_premium

    # price grid
    lo = float(min(putK * 0.6, breakeven))
    hi = float(max(callK * 1.4, breakeven))
    prices = np.linspace(lo, hi, 121)

    # RAW payoff (vectorized) — CORRECT SIGNS
    S = prices
    intrinsic_put  = np.maximum(putK - S, 0.0)   # long put → ADD
    intrinsic_call = np.maximum(S - callK, 0.0)  # short call → SUBTRACT
    pnl_per_sh = (S - data.entry_price) + intrinsic_put - intrinsic_call + net_premium
    payoff = (pnl_per_sh * data.shares)

    # HARD CLAMP: force perfectly flat tails at caps
    payoff = np.where(S <= putK,  max_loss, payoff)
    payoff = np.where(S >= callK, max_gain, payoff)

    # round/serialize
    payoff_out = [round(float(x), 2) for x in payoff.tolist()]
    prices_out = [round(float(x), 2) for x in prices.tolist()]

    return {
        "version": API_VERSION,
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
        "payoff_prices": prices_out,
        "payoff_values": payoff_out,
    }

@app.post("/debug")
def debug(data: CalcRequest):
    # call the calculate() above to reuse the exact logic
    out = calculate(data)
    pv = out["payoff_values"]
    return {
        "version": "debug",
        "selected_put_strike": out["selected_put_strike"],
        "selected_call_strike": out["selected_call_strike"],
        "net_premium": out["net_premium"],
        "max_loss": out["max_loss"],
        "max_gain": out["max_gain"],
        "breakeven_estimate": out["breakeven_estimate"],
        "payoff_first10": pv[:10],
        "payoff_last10": pv[-10:]
    }

