# main.py — FastAPI backend (v4.0: bid/ask pricing, robust spot, strikes, real Greeks, premium ping)
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
from math import log, sqrt, exp, pi
from zoneinfo import ZoneInfo

app = FastAPI()
API_VERSION = "collar-api v4.0"

# ---------------- CORS ----------------
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

# ---------------- Models ----------------
class CalcRequest(BaseModel):
    ticker: str
    shares: int
    entry_price: float
    put_strike: float
    call_strike: float
    expiration: str

# ---------------- Health ----------------
@app.get("/")
def root():
    return {"ok": True, "version": API_VERSION, "docs": "/docs", "redoc": "/redoc"}

@app.get("/health")
def health():
    return {"status": "ok", "version": API_VERSION}

@app.get("/debug")
def debug():
    return {"ok": True, "version": "debug", "ts": time.time()}


# ---------------- Time & Spot policy ----------------
ET = ZoneInfo("America/New_York")

def _now_et() -> dt.datetime:
    return dt.datetime.now(ET)

def _is_market_hours(now_et: Optional[dt.datetime] = None) -> bool:
    z = now_et or _now_et()
    if z.weekday() >= 5:
        return False
    t = z.time()
    return (t >= dt.time(9, 30)) and (t <= dt.time(16, 0))

def _last_valid(series: pd.Series) -> Optional[float]:
    try:
        v = series.dropna()
        if not v.empty:
            return float(v.iloc[-1])
    except Exception:
        pass
    return None

def _yf_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=3, backoff_factor=0.4,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s

def _spot_policy_value(tkr: yf.Ticker, fallback: Optional[float]) -> Tuple[Optional[float], str]:
    # Market hours → 1m bar (≈15m delayed). After hours → prior close.
    if _is_market_hours():
        try:
            h = tkr.history(period="5d", interval="1m")
            if isinstance(h, pd.DataFrame) and not h.empty and "Close" in h:
                v = _last_valid(h["Close"])
                if v and v > 0:
                    return v, "market_hours_15m"
        except Exception:
            pass
        try:
            fi = getattr(tkr, "fast_info", {}) or {}
            for k in ("regular_market_price", "regularMarketPrice", "last_price"):
                v = fi.get(k)
                if v and float(v) > 0:
                    return float(v), "market_hours_fastinfo"
        except Exception:
            pass
    else:
        try:
            h = tkr.history(period="5d", interval="1d")
            if isinstance(h, pd.DataFrame) and not h.empty and "Close" in h:
                v = _last_valid(h["Close"])
                if v and v > 0:
                    return v, "after_hours_close"
        except Exception:
            pass
        try:
            fi = getattr(tkr, "fast_info", {}) or {}
            for k in ("previous_close", "previousClose", "last_close"):
                v = fi.get(k)
                if v and float(v) > 0:
                    return float(v), "after_hours_fastinfo_close"
        except Exception:
            pass
    return fallback, ("fallback_entry_hint" if fallback is not None else "fallback_none")

# ---------------- Expirations & Strikes ----------------
def _us_pretty(iso: str) -> str:
    try:
        d = dt.date.fromisoformat(iso)
        return f"{d.month}/{d.day}/{d.year}"
    except Exception:
        return iso

def _next_fridays_years(years: int = 2) -> List[str]:
    out = []
    today = dt.date.today()
    days_ahead = (4 - today.weekday()) % 7
    first = today + dt.timedelta(days=days_ahead or 7)
    end = today + dt.timedelta(weeks=52 * years)
    cur = first
    while cur <= end:
        out.append(cur.isoformat())
        cur += dt.timedelta(weeks=1)
    return out

@app.get("/expirations/{ticker}")
def expirations_path(ticker: str):
    try:
        tkr = yf.Ticker(ticker, session=_yf_session())
        opts = list(tkr.options or [])
        if not opts:
            opts = _next_fridays_years(2)
        return {"symbol": ticker.upper(), "expirations": opts, "expirations_pretty": [_us_pretty(x) for x in opts]}
    except Exception:
        opts = _next_fridays_years(2)
        return {"symbol": ticker.upper(), "expirations": opts, "expirations_pretty": [_us_pretty(x) for x in opts]}

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
    tkr = yf.Ticker(symbol, session=_yf_session())
    try:
        opt = tkr.option_chain(expiration)
        calls, puts = opt.calls, opt.puts
    except Exception:
        calls, puts = pd.DataFrame(), pd.DataFrame()
    put_list = sorted({float(x) for x in (puts["strike"].tolist() if not puts.empty else [])})
    call_list = sorted({float(x) for x in (calls["strike"].tolist() if not calls.empty else [])})
    all_list = sorted(set(put_list) | set(call_list))
    return {"symbol": symbol.upper(), "expiration": expiration, "puts": put_list, "calls": call_list, "all": all_list}

# ---------------- Yahoo chain fetch (cache + fresh spot) ----------------
_CACHE_LOCK = threading.Lock()
_CHAIN_CACHE: Dict[Tuple[str, str], Dict[str, Any]] = {}

def _cache_key(ticker: str, exp: str) -> Tuple[str, str]:
    return (ticker.upper(), exp)

def _cache_put_chain(ticker: str, exp: str, calls_df: pd.DataFrame, puts_df: pd.DataFrame):
    entry = {
        "ts": time.time(),
        "calls": calls_df.to_dict("records"),
        "puts":  puts_df.to_dict("records"),
    }
    with _CACHE_LOCK:
        _CHAIN_CACHE[_cache_key(ticker, exp)] = entry

def _cache_get_chain(ticker: str, exp: str):
    with _CACHE_LOCK:
        e = _CHAIN_CACHE.get(_cache_key(ticker, exp))
    if not e:
        return None
    return {
        "calls": pd.DataFrame(e.get("calls") or []),
        "puts": pd.DataFrame(e.get("puts") or []),
        "age_sec": int(time.time() - e["ts"]),
    }

def fetch_option_chain(ticker: str, expiration: str):
    tkr = yf.Ticker(ticker, session=_yf_session())
    # find closest available expiration
    try:
        avail = list(tkr.options or [])
    except Exception:
        avail = []
    target = expiration
    if avail and expiration not in avail:
        try:
            want = dt.date.fromisoformat(expiration)
            def dist(e):
                try: return abs((dt.date.fromisoformat(e) - want).days)
                except: return 10**9
            target = min(avail, key=dist)
        except Exception:
            target = avail[0] if avail else expiration

    # live
    try:
        if target:
            opt = tkr.option_chain(target)
            calls, puts = opt.calls, opt.puts
            _cache_put_chain(ticker, target, calls, puts)
            spot, policy = _spot_policy_value(tkr, fallback=None)
            return calls, puts, spot, policy, ("yfinance" if target == expiration else f"yfinance-nearest:{target}")
    except Exception:
        pass

    # cache
    cached = _cache_get_chain(ticker, expiration)
    if cached and not cached["calls"].empty and not cached["puts"].empty:
        spot, policy = _spot_policy_value(tkr, fallback=None)
        return cached["calls"], cached["puts"], spot, policy, f"cache:{cached['age_sec']}s"

    # fallback empty
    spot, policy = _spot_policy_value(tkr, fallback=None)
    return pd.DataFrame(), pd.DataFrame(), spot, policy, "empty"

# ---------------- Pricing helpers (bid/ask & IV) ----------------
def _f(row, key):
    try:
        v = float(row.get(key, float("nan")))
        return v if v == v and v > 0 else None
    except Exception:
        return None

def _mid_from_row(row: pd.Series) -> Optional[float]:
    bid = _f(row, "bid"); ask = _f(row, "ask")
    if bid and ask:
        return (bid + ask) / 2.0
    last = _f(row, "lastPrice")
    if last:
        return last
    return bid or ask

def _price_buy(row: pd.Series) -> float:
    # Pay ask when buying
    ask = _f(row, "ask")
    if ask: return ask
    last = _f(row, "lastPrice")
    if last: return last
    mid = _mid_from_row(row)
    if mid: return mid
    bid = _f(row, "bid")
    if bid: return bid
    return 0.01

def _price_sell(row: pd.Series) -> float:
    # Receive bid when selling
    bid = _f(row, "bid")
    if bid: return bid
    last = _f(row, "lastPrice")
    if last: return last
    mid = _mid_from_row(row)
    if mid: return mid
    ask = _f(row, "ask")
    if ask: return ask
    return 0.01

def _quote_info(row: pd.Series) -> dict:
    return {
        "strike": _f(row, "strike"),
        "bid": _f(row, "bid"),
        "ask": _f(row, "ask"),
        "last": _f(row, "lastPrice"),
        "iv": _f(row, "impliedVolatility"),
    }

# ---------------- Black–Scholes & Greeks ----------------
SQRT2PI = sqrt(2*pi)
def _phi(x):  # standard normal PDF
    return math.exp(-0.5*x*x)/SQRT2PI
def _N(x):    # standard normal CDF
    return 0.5*(1.0+math.erf(x/math.sqrt(2.0)))

def _implied_vol_from_price(S,K,r,q,T,kind,price,seed=0.25):
    # Simple bounded Newton fallback
    if price is None or price<=0 or S<=0 or K<=0: return None
    sigma=max(1e-4, float(seed))
    for _ in range(30):
        d1=(log(S/K)+(r-q+0.5*sigma*sigma)*T)/(sigma*sqrt(T))
        d2=d1-sigma*sqrt(T)
        disc_q=exp(-q*T); disc_r=exp(-r*T)
        if kind=="call":
            model=S*disc_q*_N(d1)-K*disc_r*_N(d2)
        else:
            model=K*disc_r*_N(-d2)-S*disc_q*_N(-d1)
        vega=S*disc_q*_phi(d1)*sqrt(T)
        diff=model-price
        if abs(diff)<1e-4: return max(1e-4, min(5.0, sigma))
        if vega<=1e-8: break
        sigma -= diff/vega
        if sigma<1e-4: sigma=1e-4
        if sigma>5.0: sigma=5.0
    return max(1e-4, min(5.0, sigma))

def _greeks(S,K,r,q,sigma,T,kind):
    if T<=0: T=1/365
    if sigma<=0: sigma=1e-6
    d1=(log(S/K)+(r-q+0.5*sigma*sigma)*T)/(sigma*sqrt(T))
    d2=d1-sigma*sqrt(T)
    disc_q=exp(-q*T); disc_r=exp(-r*T)
    pdf=_phi(d1)
    Nd1=_N(d1); Nmd1=_N(-d1); Nd2=_N(d2); Nmd2=_N(-d2)
    delta = disc_q*Nd1 if kind=="call" else disc_q*(Nd1-1.0)    # per share
    gamma = disc_q*pdf/(S*sigma*sqrt(T))
    vega  = S*disc_q*pdf*sqrt(T)                                 # per 1.00 vol change
    theta_cont = -(S*disc_q*pdf*sigma)/(2*sqrt(T)) - (r*K*disc_r*Nd2 if kind=="call" else -r*K*disc_r*Nmd2) + q*S*disc_q*(Nd1 if kind=="call" else -Nmd1)
    theta_per_day = theta_cont/365.0
    rho   = (K*T*disc_r*Nd2 if kind=="call" else -K*T*disc_r*Nmd2)
    return {"delta":delta,"gamma":gamma,"vega":vega,"theta":theta_per_day,"rho":rho}

# ---------------- Helpers to pick chain rows ----------------
def _pick_row_exact(df: pd.DataFrame, strike: float) -> Optional[pd.Series]:
    if df is None or df.empty: return None
    m = df[df["strike"] == strike]
    return None if m.empty else m.iloc[0]

def _pick_row_nearest(df: pd.DataFrame, strike: float) -> Optional[pd.Series]:
    if df is None or df.empty: return None
    try:
        s = df["strike"].astype(float)
        idx = (s - float(strike)).abs().idxmin()
        return df.loc[idx]
    except Exception:
        return None

# ---------------- /calculate ----------------
@app.post("/calculate")
def calculate(data: CalcRequest):
    tkr = data.ticker.upper()
    calls, puts, spot, spot_policy, source = fetch_option_chain(tkr, data.expiration)

    # entry = spot per policy (fallback to provided hint)
    entry_used = float(spot) if spot is not None else float(data.entry_price)

    # find rows
    put_row  = _pick_row_exact(puts, data.put_strike)  or _pick_row_nearest(puts,  data.put_strike)
    call_row = _pick_row_exact(calls, data.call_strike) or _pick_row_nearest(calls, data.call_strike)
    if put_row is None or call_row is None:
        raise HTTPException(status_code=400, detail="Strikes not found. Use /strikes to list available strikes.")

    # BUY put (ask), SELL call (bid)
    put_price  = _price_buy(put_row)
    call_price = _price_sell(call_row)
    net_premium = call_price - put_price

    # $1 grid payoff; straight line segments (front-end: remove stepped:true)
    lo_ref = min(data.put_strike, entry_used, spot if spot is not None else entry_used)
    hi_ref = max(data.call_strike, entry_used, spot if spot is not None else entry_used)
    lo = int(max(1, math.floor(lo_ref * 0.75))); hi = int(math.ceil(hi_ref * 1.25))
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

    return {
        "ticker": tkr,
        "spot_price": float(entry_used),
        "spot_policy": spot_policy,
        "data_source": source,
        "entry_price_requested": float(data.entry_price),
        "entry_price_used": float(entry_used),
        "shares": data.shares,
        "expiration": data.expiration,
        "selected_put_strike": float(data.put_strike),
        "selected_call_strike": float(data.call_strike),
        "put_premium_paid": round(put_price, 3),          # ask
        "call_premium_received": round(call_price, 3),    # bid
        "net_premium": round(net_premium, 3),
        "max_loss": max_loss,
        "max_gain": max_gain,
        "breakeven_estimate": round(entry_used + net_premium, 3),
        "payoff_prices": [float(x) for x in prices],
        "payoff_values": [float(x) for x in payoff_arr.tolist()],
        "quotes": { "put": _quote_info(put_row), "call": _quote_info(call_row) },
    }

# ---------------- Premium: ping & calculate ----------------
@app.get("/premium/ping")
def premium_ping(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-KEY"),
    x_premium_key: Optional[str] = Header(default=None, alias="X-Premium-Key"),
    api_key: Optional[str] = Query(default=None),
    premium_key: Optional[str] = Query(default=None),
):
    expected = os.environ.get("PREMIUM_API_KEY")
    provided = x_api_key or x_premium_key or api_key or premium_key
    return {"ok": bool(expected), "matched": bool(expected and provided == expected)}

@app.post("/premium/calculate")
def premium_calculate(
    payload: Dict[str, Any] = Body(...),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-KEY"),
    x_premium_key: Optional[str] = Header(default=None, alias="X-Premium-Key"),
    api_key_q: Optional[str] = Query(default=None),
    premium_key_q: Optional[str] = Query(default=None),
):
    expected = os.environ.get("PREMIUM_API_KEY")
    provided = x_api_key or x_premium_key or api_key_q or premium_key_q or payload.get("api_key") or payload.get("premium_key")
    if expected and provided != expected:
        raise HTTPException(status_code=403, detail="Forbidden")

    # Build CalcRequest robustly (no pydantic v1/v2 dependency)
    fields = ("ticker","shares","entry_price","put_strike","call_strike","expiration")
    try:
        data = CalcRequest(**{k: payload[k] for k in fields})
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing field: {e}")

    # base calc (gets us spot, quotes, payoff)
    base = calculate(data)

    # prepare for Greeks
    tkr = yf.Ticker(data.ticker, session=_yf_session())
    S = float(base["spot_price"])
    try:
        exp_dt = dt.date.fromisoformat(data.expiration)
        days = max((exp_dt - dt.date.today()).days, 1)
    except Exception:
        days = 30
    T = days/365.0
    fi = getattr(tkr, "fast_info", {}) or {}
    q = float(fi.get("dividend_yield") or 0.0) or 0.0
    r = 0.045
    contracts = data.shares / 100.0   # 1 option per 100 shares
    multiplier = 100.0

    # fetch rows again for IVs
    calls, puts, _, _, _ = fetch_option_chain(data.ticker, data.expiration)
    prow = _pick_row_exact(puts, data.put_strike)  or _pick_row_nearest(puts,  data.put_strike)
    crow = _pick_row_exact(calls, data.call_strike) or _pick_row_nearest(calls, data.call_strike)

    # pick IVs (prefer Yahoo IV; else derive from price; else fallback)
    iv_put = (prow.get("impliedVolatility") if isinstance(prow, pd.Series) else None)
    iv_call = (crow.get("impliedVolatility") if isinstance(crow, pd.Series) else None)
    if iv_put is None:
        put_pay = _price_buy(prow) if isinstance(prow, pd.Series) else None
        iv_put = _implied_vol_from_price(S, data.put_strike, r, q, T, "put", put_pay) or 0.25
    if iv_call is None:
        call_rcv = _price_sell(crow) if isinstance(crow, pd.Series) else None
        iv_call = _implied_vol_from_price(S, data.call_strike, r, q, T, "call", call_rcv) or 0.25
    iv_put = float(iv_put); iv_call = float(iv_call)

    # leg greeks per share
    g_put  = _greeks(S, data.put_strike,  r, q, iv_put,  T, "put")
    g_call = _greeks(S, data.call_strike, r, q, iv_call, T, "call")

    # portfolio: +shares*1 + long puts (contracts*100) – short calls (contracts*100)
    net = {k:0.0 for k in ("delta","gamma","vega","theta","rho")}
    # stock
    net["delta"] += data.shares * 1.0
    # put (long)
    for k,v in g_put.items():
        net[k] += v * (contracts * multiplier)
    # call (short)
    for k,v in g_call.items():
        net[k] -= v * (contracts * multiplier)

    # per-share view (helpful in some UIs)
    per_share = {k: net[k]/max(1.0, data.shares) for k in net}

    # AnchorLock placeholders (deterministic)
    rng = np.random.default_rng(seed=(abs(hash((data.ticker, data.expiration))) % 2_147_483_647))
    rsi = round(float(rng.uniform(20, 80)), 2)
    momentum = round(float(rng.uniform(-1, 1)), 3)
    earnings = round(float(rng.uniform(0, 100)), 1)
    ma200 = round(float(rng.uniform(150, 300)), 2)
    gap200 = round(((S - ma200)/ma200 if ma200 else 0.0), 4)
    slope200 = round(float(rng.uniform(-2, 2)), 3)

    score = 50.0 + (rsi-50.0)*0.5 + momentum*25.0 + (earnings-50.0)*0.05
    score = max(0.0, min(100.0, score))
    action = "WATCH"
    if score >= 70.0: action = "ROLL DOWN/CAP"
    elif score <= 30.0: action = "ROLL UP/FLOOR"

    # assemble payload
    base["iv"] = {"put": iv_put, "call": iv_call}
    base["iv_put"] = iv_put
    base["iv_call"] = iv_call

    base["portfolioGreeks"] = {
        "net": {
            "Delta": round(net["delta"], 4),
            "Gamma": round(net["gamma"], 6),
            "Vega":  round(net["vega"], 4),
            "Theta": round(net["theta"], 4),  # per day
            "Rho":   round(net["rho"], 4),
            # lowercase mirrors
            "delta": round(net["delta"], 4),
            "gamma": round(net["gamma"], 6),
            "vega":  round(net["vega"], 4),
            "theta": round(net["theta"], 4),
            "rho":   round(net["rho"], 4),
        },
        "perShare": {k: round(v,6) for k,v in per_share.items()},
        "contracts_used": contracts,
        "multiplier": multiplier,
    }
    # top-level greeks (aliases, per-share)
    base["greeks"] = {k: round(v, 6 if k=="gamma" else 4) for k,v in per_share.items()}
    for k,v in base["greeks"].items():
        base[k] = v

    base["components"] = {
        "RSI": rsi,
        "RSIScore": rsi,
        "Momentum30d": momentum,
        "DMA200": ma200,
        "GapTo200DMA": gap200,
        "DMA200Slope30d": slope200,
        "EarningsScore": earnings,
    }
    base["assumptions"] = {
        "r": r, "q": q,
        "time_to_exp_years": round(T, 4),
        "contracts": {"stock": data.shares, "put": contracts, "call": contracts},
    }
    base["signals"] = {"score": round(score,1), "action": action}
    base["signal"] = action

    return base

# ---------------- GET helper ----------------
@app.get("/calculate")
def calculate_get_help():
    return {
        "error": "Use POST /calculate with JSON.",
        "spot_policy": "15m last bar during US market hours; prior close after hours",
        "example_payload": {
            "ticker": "AAPL", "shares": 100, "entry_price": 190,
            "put_strike": 175, "call_strike": 205, "expiration": "2026-01-16"
        }
    }
