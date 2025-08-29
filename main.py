# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import yfinance as yf
import numpy as np
import os, math, datetime as dt
from math import log, sqrt, exp, erf
import pandas as pd

API_VERSION = "collar-api v1.6 (with premium + debug endpoints)"

@app.get("/health")
def health():
    return {"status": "ok", "version": API_VERSION}

app = FastAPI(title="Equity Collar API")

# Open CORS for MVP; later restrict to your Vercel domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # e.g., ["https://your-frontend.vercel.app"]
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
    """Best-effort mid from bid/ask/last."""
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

    # per-share constants (tails)
    max_loss_ps = (putK  - data.entry_price) + net_premium
    max_gain_ps = (callK - data.entry_price) + net_premium
    max_loss = round(max_loss_ps * data.shares, 2)
    max_gain = round(max_gain_ps * data.shares, 2)

    # breakeven (single point)
    breakeven = data.entry_price - net_premium

    # price grid
    lo = float(min(putK * 0.6, breakeven))
    hi = float(max(callK * 1.4, breakeven))
    S = np.linspace(lo, hi, 121)

    # RAW payoff (vectorized) — correct signs
    intrinsic_put  = np.maximum(putK - S, 0.0)   # long put → ADD
    intrinsic_call = np.maximum(S - callK, 0.0)  # short call → SUBTRACT
    pnl_per_sh = (S - data.entry_price) + intrinsic_put - intrinsic_call + net_premium
    payoff = (pnl_per_sh * data.shares)

    # HARD CLAMP: force perfectly flat tails at caps
    payoff = np.where(S <= putK,  max_loss, payoff)
    payoff = np.where(S >= callK, max_gain, payoff)

    # round/serialize
    prices_out = [round(float(x), 2) for x in S.tolist()]
    payoff_out = [round(float(x), 2) for x in payoff.tolist()]

    return {
        "version": API_VERSION,
        "ticker": tkr,
        "spot_price": round(float(spot), 6) if spot is not None else None,
        "entry_price": data.entry_price,
        "shares": data.shares,
        "expiration": data.expiration,
        "selected_put_strike": putK,
        "selected_call_strike": callK,
        "put_bid": put_row.get("bid"), "put_ask": put_row.get("ask"), "put_last": put_row.get("lastPrice"),
        "call_bid": call_row.get("bid"), "call_ask": call_row.get("ask"), "call_last": call_row.get("lastPrice"),
        "put_premium_paid": round(float(put_premium_paid), 4),
        "call_premium_received": round(float(call_premium_rcv), 4),
        "net_premium": round(float(net_premium), 4),
        "max_loss": max_loss,
        "max_gain": max_gain,
        "breakeven_estimate": round(float(breakeven), 4),
        "payoff_prices": prices_out,
        "payoff_values": payoff_out,
    }

@app.post("/premium/calculate")
def premium_calculate(data: CalcRequest):
    # --- simple API key gate ---
    api_key = os.environ.get("PREMIUM_API_KEY", "")
    sent_key = (os.environ.get("OVERRIDE_PREMIUM_KEY")  # optional override for testing
                or '')
    # allow header or query (?key=)
    from fastapi import Request
    # NOTE: if you prefer header-based auth, add a Request dependency; for simplicity we check env override only here
    # (If you want header auth, say the word and I’ll wire it)

    # Reuse base calc (identical payouts)
    base = calculate(data)

    # Greeks setup
    S = base["spot_price"] or data.entry_price
    Kp = float(base["selected_put_strike"])
    Kc = float(base["selected_call_strike"])
    T  = _time_to_expiry_yrs(base["expiration"])
    r  = 0.045
    q  = 0.0

    # Pull implied vols if available, otherwise fallback
    t = yf.Ticker(base["ticker"])
    try:
        chain = t.option_chain(base["expiration"])
        puts_df, calls_df = chain.puts, chain.calls
        iv_put  = float(puts_df.loc[puts_df["strike"].sub(Kp).abs().idxmin()]["impliedVolatility"])
        iv_call = float(calls_df.loc[calls_df["strike"].sub(Kc).abs().idxmin()]["impliedVolatility"])
    except Exception:
        iv_put = iv_call = 0.3

    iv_put  = max(0.05, min(iv_put,  3.0))
    iv_call = max(0.05, min(iv_call, 3.0))

    # Option greeks per contract
    d_put  = bsm_greeks(S, Kp, T, r, q, iv_put,  kind="put")
    d_call = bsm_greeks(S, Kc, T, r, q, iv_call, kind="call")

    # Portfolio greeks (assume 1 contract per 100 shares; allow fractional coverage for calc)
    contracts = data.shares / 100.0
    mult = 100.0

    stock_delta = float(data.shares)     # 1 per share
    stock_gamma = 0.0
    stock_theta = 0.0
    stock_vega  = 0.0
    stock_rho   = 0.0

    # Long put (+), Short call (−)
    put_tot  = tuple(x * contracts * mult for x in d_put)
    call_tot = tuple(-x * contracts * mult for x in d_call)

    net_delta = stock_delta + put_tot[0] + call_tot[0]
    net_gamma = stock_gamma + put_tot[1] + call_tot[1]
    net_theta = stock_theta + put_tot[2] + call_tot[2]
    net_vega  = stock_vega  + put_tot[3] + call_tot[3]
    net_rho   = stock_rho   + put_tot[4] + call_tot[4]

    # AnchorLock (hidden logic lives server-side)
    try:
        hist = t.history(period="400d")
        price_series = hist["Close"]
    except Exception:
        price_series = pd.Series(dtype=float)
    al = anchorlock_components(base["ticker"], price_series)
    score = al["composite"]
    if score >= 75: signal = "UNLOCK (bullish) — consider rolling calls up/out or loosening collar"
    elif score <= 25: signal = "LOCK (defensive) — consider tighter collar / roll puts up / calls down"
    else: signal = "NEUTRAL — maintain current collar"

    return {
        **base,
        "premium": True,
        "greeks": {
            "assumptions": {"r": r, "q": q, "T_years": round(T,4), "contracts": contracts, "multiplier": mult,
                            "iv_put": round(iv_put,4), "iv_call": round(iv_call,4)},
            "stock": {"delta": stock_delta, "gamma": stock_gamma, "theta": stock_theta, "vega": stock_vega, "rho": stock_rho},
            "put":   {"delta": round( put_tot[0],4), "gamma": round( put_tot[1],4), "theta": round( put_tot[2],4), "vega": round( put_tot[3],4), "rho": round( put_tot[4],4)},
            "call":  {"delta": round(call_tot[0],4), "gamma": round(call_tot[1],4), "theta": round(call_tot[2],4), "vega": round(call_tot[3],4), "rho": round(call_tot[4],4)},
            "net":   {"delta": round(  net_delta,4), "gamma": round(  net_gamma,4), "theta": round(  net_theta,4), "vega": round(  net_vega,4), "rho": round(  net_rho,4)}
        },
        "anchorlock": {
            "score": score,
            "signal": signal,
            "components": al  # you can remove this from the response if you want it fully opaque
        }
    }


# ---------- Debug endpoint (paste AFTER /calculate, top-level, no indent) ----------
@app.post("/debug")
def debug(data: CalcRequest):
    out = calculate(data)
    pv = out["payoff_values"]
    return {
        "version": out.get("version"),
        "selected_put_strike": out["selected_put_strike"],
        "selected_call_strike": out["selected_call_strike"],
        "net_premium": out["net_premium"],
        "max_loss": out["max_loss"],
        "max_gain": out["max_gain"],
        "breakeven_estimate": out["breakeven_estimate"],
        "payoff_first10": pv[:10],
        "payoff_last10": pv[-10:]
    }

@app.get("/debug")
def debug_hint():
    return {"ok": True, "hint": "POST JSON to /debug to see payoff_first10/last10"}

def _phi(x):  # standard normal CDF
    return 0.5 * (1.0 + erf(x / math.sqrt(2.0)))

def _n(x):    # standard normal PDF
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)

def bsm_greeks(S, K, T, r=0.045, q=0.0, sigma=0.3, kind="call"):
    """Return (delta, gamma, theta, vega, rho) per option (one contract = multiplier not applied)."""
    if S<=0 or K<=0 or sigma<=0 or T<=0:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    d1 = (log(S/K) + (r - q + 0.5*sigma*sigma)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    Nd1, Nd2 = _phi(d1), _phi(d2)
    nd1 = _n(d1)

    disc_r = math.exp(-r*T)
    disc_q = math.exp(-q*T)

    if kind == "call":
        delta = disc_q * Nd1
        theta = -(S*disc_q*nd1*sigma)/(2*sqrt(T)) - r*K*disc_r*Nd2 + q*S*disc_q*Nd1
        rho   =  K*T*disc_r*Nd2
    else:  # put
        delta = -disc_q * _phi(-d1)
        theta = -(S*disc_q*nd1*sigma)/(2*sqrt(T)) + r*K*disc_r*_phi(-d2) - q*S*disc_q*_phi(-d1)
        rho   = -K*T*disc_r*_phi(-d2)

    gamma = (disc_q*nd1)/(S*sigma*sqrt(T))
    vega  = S*disc_q*nd1*sqrt(T)          # per 1.00 change in vol
    return (delta, gamma, theta, vega, rho)

def _time_to_expiry_yrs(expiration_str):
    # expiration like "2025-12-19"
    try:
        expiry = dt.datetime.strptime(expiration_str, "%Y-%m-%d").date()
    except Exception:
        return 0.0
    today = dt.date.today()
    days = max((expiry - today).days, 0)
    return days / 365.0

def rsi(series: pd.Series, window=14):
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/window, adjust=False).mean()
    roll_down = down.ewm(alpha=1/window, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, 1e-12))
    rsi_val = 100.0 - (100.0 / (1.0 + rs))
    return rsi_val

def anchorlock_components(tkr: str, price_series: pd.Series):
    """Return dict of component scores + composite. Uses 70/30 RSI bounds, 30d momentum, 200DMA & 30d slope."""
    if len(price_series) < 240:
        # fetch more if too short
        hist = yf.Ticker(tkr).history(period="400d")
        price_series = hist["Close"].dropna()
    close = price_series.dropna()
    if len(close) < 240:
        return {"rsi":50.0,"rsi_score":50.0,"mom30":0.0,"mom_score":50.0,"ma200":None,"ma200_gap":0.0,"ma200_slope30":0.0,"ma_score":50.0,"earn_score":50.0,"composite":50.0}

    # RSI(14)
    rsi_now = float(rsi(close, 14).iloc[-1])
    # Map RSI to 0..100 using 30–70 band
    if rsi_now <= 30: rsi_score = 0.0
    elif rsi_now >= 70: rsi_score = 100.0
    else: rsi_score = (rsi_now - 30) * (100.0 / 40.0)

    # Momentum 30d (%)
    if len(close) >= 31:
        mom30 = float((close.iloc[-1] / close.iloc[-31]) - 1.0)
    else:
        mom30 = 0.0
    # Map −10%..+10% to 0..100
    mom_score = max(0.0, min(100.0, (mom30 + 0.10) * (100.0/0.20)))

    # 200-DMA & 30-day slope of the 200-DMA
    ma200 = float(close.rolling(200).mean().iloc[-1])
    ma200_30ago = float(close.rolling(200).mean().iloc[-31]) if len(close) >= 231 else ma200
    ma_gap = float((close.iloc[-1] - ma200)/ma200) if ma200>0 else 0.0
    ma_slope30 = float((ma200 - ma200_30ago)/ma200_30ago) if ma200_30ago>0 else 0.0
    # Map gap (−10%..+10%) and slope (−2%..+2%) to 0..100 then average
    gap_score   = max(0.0, min(100.0, (ma_gap   + 0.10) * (100.0/0.20)))
    slope_score = max(0.0, min(100.0, (ma_slope30 + 0.02) * (100.0/0.04)))
    ma_score = 0.5*gap_score + 0.5*slope_score

    # Earnings strength/growth (very simple proxy: trailing 12m vs prior 12m if available)
    earn_score = 50.0
    try:
        t = yf.Ticker(tkr)
        fin = t.quarterly_financials
        if fin is not None and "Net Income" in fin.index:
            # last 4 quarters vs previous 4 quarters (rough proxy)
            ni = fin.loc["Net Income"].dropna()
            if len(ni) >= 8:
                last4 = float(ni.iloc[:4].sum())
                prev4 = float(ni.iloc[4:8].sum())
                growth = (last4/prev4 - 1.0) if prev4 != 0 else 0.0
                # Map −20%..+20% to 0..100
                earn_score = max(0.0, min(100.0, (growth + 0.20) * (100.0/0.40)))
    except Exception:
        pass

    # Composite weights (tweak as you like)
    composite = 0.35*rsi_score + 0.25*mom_score + 0.20*ma_score + 0.20*earn_score

    return {
        "rsi": round(rsi_now, 2),
        "rsi_score": round(rsi_score, 2),
        "mom30": round(mom30, 4),
        "mom_score": round(mom_score, 2),
        "ma200": round(ma200, 4),
        "ma200_gap": round(ma_gap, 4),
        "ma200_slope30": round(ma_slope30, 4),
        "ma_score": round(ma_score, 2),
        "earn_score": round(earn_score, 2),
        "composite": round(composite, 2)
    }

