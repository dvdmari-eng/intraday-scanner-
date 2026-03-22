from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf


@dataclass
class DecisionResult:
    symbol: str
    last_price: float
    score: int
    decision: str
    reasons: list
    snapshot: Dict[str, float]


def download_data(symbol: str, period: str = "10d", interval: str = "5m") -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for {symbol}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    needed = ["Open", "High", "Low", "Close", "Volume"]
    df = df[needed].copy().dropna()
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["MA20"] = out["Close"].rolling(20).mean()
    out["MA50"] = out["Close"].rolling(50).mean()
    out["MA200"] = out["Close"].rolling(200).mean()

    delta = out["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["RSI14"] = 100 - (100 / (1 + rs))

    out["VOL_AVG20"] = out["Volume"].rolling(20).mean()
    out["RVOL"] = out["Volume"] / out["VOL_AVG20"]

    out["RANGE"] = (out["High"] - out["Low"]).replace(0, np.nan)
    out["CLOSE_POS"] = (out["Close"] - out["Low"]) / out["RANGE"]

    idx_naive = out.index.tz_localize(None) if out.index.tz is not None else out.index
    out["SESSION"] = idx_naive.date

    typical_price = (out["High"] + out["Low"] + out["Close"]) / 3
    out["TPV"] = typical_price * out["Volume"]
    out["CUM_TPV"] = out.groupby("SESSION")["TPV"].cumsum()
    out["CUM_VOL"] = out.groupby("SESSION")["Volume"].cumsum().replace(0, np.nan)
    out["VWAP"] = out["CUM_TPV"] / out["CUM_VOL"]

    out["PREV5_HIGH"] = out["High"].shift(1).rolling(5).max()
    out["PREV5_LOW"] = out["Low"].shift(1).rolling(5).min()

    out = out.drop(columns=["TPV", "CUM_TPV", "CUM_VOL"])
    return out


def detect_cross(prev_fast: float, prev_slow: float, now_fast: float, now_slow: float) -> Tuple[bool, bool]:
    cross_up = (
        pd.notna(prev_fast)
        and pd.notna(prev_slow)
        and pd.notna(now_fast)
        and pd.notna(now_slow)
        and prev_fast <= prev_slow
        and now_fast > now_slow
    )

    cross_down = (
        pd.notna(prev_fast)
        and pd.notna(prev_slow)
        and pd.notna(now_fast)
        and pd.notna(now_slow)
        and prev_fast >= prev_slow
        and now_fast < now_slow
    )

    return cross_up, cross_down


def score_setup(df: pd.DataFrame, symbol: str) -> DecisionResult:
    if len(df) < 220:
        raise ValueError(f"Not enough bars to calculate MA200 reliably for {symbol}")

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    score = 0
    reasons = []

    if latest["Close"] > latest["VWAP"]:
        score += 2
        reasons.append("Price is above VWAP (+2)")
    elif latest["Close"] < latest["VWAP"]:
        score -= 2
        reasons.append("Price is below VWAP (-2)")

    if latest["MA20"] > latest["MA50"]:
        score += 1
        reasons.append("MA20 is above MA50 (+1)")
    elif latest["MA20"] < latest["MA50"]:
        score -= 1
        reasons.append("MA20 is below MA50 (-1)")

    if latest["MA50"] > latest["MA200"]:
        score += 1
        reasons.append("MA50 is above MA200 (+1)")
    elif latest["MA50"] < latest["MA200"]:
        score -= 1
        reasons.append("MA50 is below MA200 (-1)")

    cross_up_20_50, cross_down_20_50 = detect_cross(prev["MA20"], prev["MA50"], latest["MA20"], latest["MA50"])
    if cross_up_20_50:
        score += 1
        reasons.append("MA20 crossed above MA50 now (+1)")
    elif cross_down_20_50:
        score -= 1
        reasons.append("MA20 crossed below MA50 now (-1)")

    cross_up_50_200, cross_down_50_200 = detect_cross(prev["MA50"], prev["MA200"], latest["MA50"], latest["MA200"])
    if cross_up_50_200:
        score += 2
        reasons.append("MA50 crossed above MA200 now (+2)")
    elif cross_down_50_200:
        score -= 2
        reasons.append("MA50 crossed below MA200 now (-2)")

    if pd.notna(latest["RVOL"]) and latest["RVOL"] >= 1.5:
        if latest["Close"] > latest["Open"]:
            score += 1
            reasons.append("Strong bullish volume (RVOL >= 1.5 on green candle) (+1)")
        elif latest["Close"] < latest["Open"]:
            score -= 1
            reasons.append("Strong bearish volume (RVOL >= 1.5 on red candle) (-1)")

    if pd.notna(latest["CLOSE_POS"]):
        if latest["CLOSE_POS"] >= 0.66:
            score += 1
            reasons.append("Candle closed in upper third of range (+1)")
        elif latest["CLOSE_POS"] <= 0.33:
            score -= 1
            reasons.append("Candle closed in lower third of range (-1)")

    if pd.notna(latest["PREV5_HIGH"]) and latest["Close"] > latest["PREV5_HIGH"]:
        score += 1
        reasons.append("Breakout above previous 5-bar high (+1)")
    elif pd.notna(latest["PREV5_LOW"]) and latest["Close"] < latest["PREV5_LOW"]:
        score -= 1
        reasons.append("Breakdown below previous 5-bar low (-1)")

    if pd.notna(latest["RSI14"]):
        if latest["RSI14"] >= 60:
            score += 1
            reasons.append("RSI14 is strong (>= 60) (+1)")
        elif latest["RSI14"] <= 40:
            score -= 1
            reasons.append("RSI14 is weak (<= 40) (-1)")

    if score >= 6:
        decision = "STRONG LONG"
    elif score >= 3:
        decision = "LONG SETUP"
    elif score <= -6:
        decision = "STRONG SHORT"
    elif score <= -3:
        decision = "SHORT SETUP"
    else:
        decision = "WAIT"

    latest_ts = df.index[-1]
    latest_ts = latest_ts.tz_convert("UTC") if getattr(latest_ts, "tzinfo", None) else latest_ts

    snapshot = {
        "close": float(latest["Close"]),
        "open": float(latest["Open"]),
        "high": float(latest["High"]),
        "low": float(latest["Low"]),
        "vwap": float(latest["VWAP"]),
        "ma20": float(latest["MA20"]),
        "ma50": float(latest["MA50"]),
        "ma200": float(latest["MA200"]),
        "rsi14": float(latest["RSI14"]),
        "volume": float(latest["Volume"]),
        "vol_avg20": float(latest["VOL_AVG20"]),
        "rvol": float(latest["RVOL"]),
        "close_pos": float(latest["CLOSE_POS"]),
        "last_bar_time": str(latest_ts),
    }

    return DecisionResult(
        symbol=symbol,
        last_price=float(latest["Close"]),
        score=score,
        decision=decision,
        reasons=reasons,
        snapshot=snapshot,
    )


def score_explanation(score: float) -> str:
    if pd.isna(score):
        return "No score available."
    if score >= 6:
        return "Very strong bullish setup based on your rules."
    if score >= 3:
        return "Positive setup, but not the strongest level."
    if score <= -6:
        return "Very strong bearish setup based on your rules."
    if score <= -3:
        return "Negative setup, but not the strongest level."
    return "Mixed signals. No strong edge right now."


def decision_explanation(decision: str) -> str:
    explanations = {
        "STRONG LONG": "Most signals align to the upside. Strong bullish intraday setup.",
        "LONG SETUP": "Bullish bias exists, but confirmation is weaker than strong long.",
        "WAIT": "Signals are mixed or weak. Better to wait for clearer structure.",
        "SHORT SETUP": "Bearish bias exists, but confirmation is weaker than strong short.",
        "STRONG SHORT": "Most signals align to the downside. Strong bearish intraday setup.",
    }
    return explanations.get(decision, "No explanation available.")


def analyze_symbol(symbol: str) -> Dict:
    raw = download_data(symbol=symbol, period="10d", interval="5m")
    enriched = add_indicators(raw)
    result = score_setup(enriched, symbol)

    return {
        "Symbol": result.symbol,
        "Price": round(result.last_price, 2),
        "Score": result.score,
        "Decision": result.decision,
        "VWAP": round(result.snapshot["vwap"], 2),
        "MA20": round(result.snapshot["ma20"], 2),
        "MA50": round(result.snapshot["ma50"], 2),
        "MA200": round(result.snapshot["ma200"], 2),
        "RSI14": round(result.snapshot["rsi14"], 2),
        "RVOL": round(result.snapshot["rvol"], 2),
        "Close_vs_VWAP": round(result.snapshot["close"] - result.snapshot["vwap"], 2),
        "Last Bar": result.snapshot["last_bar_time"],
        "Reasons": " | ".join(result.reasons[:4]),
    }


def analyze_watchlist(symbols: List[str]) -> pd.DataFrame:
    rows = []

    for symbol in symbols:
        symbol = symbol.strip().upper()
        if not symbol:
            continue
        try:
            rows.append(analyze_symbol(symbol))
        except Exception as e:
            rows.append(
                {
                    "Symbol": symbol,
                    "Price": np.nan,
                    "Score": np.nan,
                    "Decision": f"ERROR: {str(e)}",
                    "VWAP": np.nan,
                    "MA20": np.nan,
                    "MA50": np.nan,
                    "MA200": np.nan,
                    "RSI14": np.nan,
                    "RVOL": np.nan,
                    "Close_vs_VWAP": np.nan,
                    "Last Bar": "",
                    "Reasons": "",
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    decision_order = {
        "STRONG LONG": 5,
        "LONG SETUP": 4,
        "WAIT": 3,
        "SHORT SETUP": 2,
        "STRONG SHORT": 1,
    }
    df["DecisionRank"] = df["Decision"].map(decision_order).fillna(0)
    df = df.sort_values(by=["Score", "DecisionRank"], ascending=[False, False]).drop(columns=["DecisionRank"])
    return df


def decision_color(decision: str) -> str:
    if decision == "STRONG LONG":
        return "#0f9d58"
    if decision == "LONG SETUP":
        return "#34a853"
    if decision == "WAIT":
        return "#fbbc05"
    if decision == "SHORT SETUP":
        return "#ea4335"
    if decision == "STRONG SHORT":
        return "#b31412"
    return "#999999"


def style_decision_table(df: pd.DataFrame):
    def color_row(row):
        color = decision_color(row["Decision"])
        return [f"background-color: {color}; color: white" if col == "Decision" else "" for col in row.index]
    return df.style.apply(color_row, axis=1)


def build_candlestick_chart(chart_df: pd.DataFrame, symbol: str):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
    )

    fig.add_trace(
        go.Candlestick(
            x=chart_df.index,
            open=chart_df["Open"],
            high=chart_df["High"],
            low=chart_df["Low"],
            close=chart_df["Close"],
            name="Candles",
            increasing_line_color="green",
            increasing_fillcolor="green",
            decreasing_line_color="red",
            decreasing_fillcolor="red",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=chart_df.index,
            y=chart_df["MA20"],
            mode="lines",
            name="MA20",
            line=dict(color="#39ff14", width=2),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=chart_df.index,
            y=chart_df["MA50"],
            mode="lines",
            name="MA50",
            line=dict(color="blue", width=2),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=chart_df.index,
            y=chart_df["MA200"],
            mode="lines",
            name="MA200",
            line=dict(color="yellow", width=2),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=chart_df.index,
            y=chart_df["VWAP"],
            mode="lines",
            name="VWAP",
            line=dict(color="red", width=2),
        ),
        row=1,
        col=1,
    )

    volume_colors = [
        "green" if close >= open_ else "red"
        for open_, close in zip(chart_df["Open"], chart_df["Close"])
    ]

    fig.add_trace(
        go.Bar(
            x=chart_df.index,
            y=chart_df["Volume"],
            name="Volume",
            marker_color=volume_colors,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=f"{symbol} - 5 Minute Candlestick Chart",
        xaxis_rangeslider_visible=False,
        height=750,
        legend_title="Indicators",
        template="plotly_dark",
        margin=dict(l=20, r=20, t=50, b=20),
    )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    return fig


st.set_page_config(page_title="Intraday 5M Trading Dashboard", layout="wide")

st.title("Intraday 5-Minute Trading Dashboard")
st.caption("VWAP + MA20/50/200 + RSI + RVOL + 5-bar breakout/breakdown")

with st.sidebar:
    st.header("Settings")
    watchlist_text = st.text_area(
        "Symbols (comma separated)",
        value="AAPL,NVDA,TSLA,AMD,META",
        height=120,
    )
    refresh = st.checkbox("Auto refresh", value=False)
    refresh_seconds = st.slider("Refresh every X seconds", min_value=30, max_value=300, value=60, step=30)
    top_n = st.slider("Top N Results", 3, 20, 5)
    scan_button = st.button("Scan now", type="primary")

symbols = [s.strip().upper() for s in watchlist_text.split(",") if s.strip()]

if refresh:
    st.markdown(
        f"<meta http-equiv='refresh' content='{refresh_seconds}'>",
        unsafe_allow_html=True,
    )

if scan_button or refresh or "scan_results" not in st.session_state:
    with st.spinner("Scanning watchlist..."):
        st.session_state["scan_results"] = analyze_watchlist(symbols)

scan_df = st.session_state.get("scan_results", pd.DataFrame())

if scan_df.empty:
    st.warning("No valid symbols to analyze yet.")
    st.stop()

display_df = scan_df.copy()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Strong Long", int((display_df["Decision"] == "STRONG LONG").sum()))
with col2:
    st.metric("Long Setup", int((display_df["Decision"] == "LONG SETUP").sum()))
with col3:
    st.metric("Short Setup", int((display_df["Decision"] == "SHORT SETUP").sum()))
with col4:
    st.metric("Strong Short", int((display_df["Decision"] == "STRONG SHORT").sum()))

st.subheader("Watchlist Analysis")

visible_columns = [
    "Symbol",
    "Price",
    "Score",
    "Decision",
    "VWAP",
    "MA20",
    "MA50",
    "MA200",
    "RSI14",
    "RVOL",
    "Close_vs_VWAP",
    "Last Bar",
    "Reasons",
]

st.dataframe(
    style_decision_table(display_df[visible_columns]),
    use_container_width=True,
    height=420,
)

st.markdown("### Score / Decision Help")
help_col1, help_col2 = st.columns(2)

with help_col1:
    st.info(
        "**Score**\n\n"
        "- 6 or more: very strong bullish alignment\n"
        "- 3 to 5: bullish setup\n"
        "- -2 to 2: mixed / wait\n"
        "- -3 to -5: bearish setup\n"
        "- -6 or less: very strong bearish alignment"
    )

with help_col2:
    st.info(
        "**Decision**\n\n"
        "- STRONG LONG: strongest bullish setup\n"
        "- LONG SETUP: bullish but weaker\n"
        "- WAIT: mixed signals\n"
        "- SHORT SETUP: bearish but weaker\n"
        "- STRONG SHORT: strongest bearish setup"
    )

longs = display_df[display_df["Decision"].isin(["STRONG LONG", "LONG SETUP"])]
shorts = display_df[display_df["Decision"].isin(["STRONG SHORT", "SHORT SETUP"])]

left, right = st.columns(2)
with left:
    st.subheader("Top Long Candidates")
    if longs.empty:
        st.info("No long candidates right now.")
    else:
        st.dataframe(style_decision_table(longs.head(top_n)), use_container_width=True)

with right:
    st.subheader("Top Short Candidates")
    if shorts.empty:
        st.info("No short candidates right now.")
    else:
        st.dataframe(style_decision_table(shorts.head(top_n)), use_container_width=True)

st.subheader("Symbol Details")
selected_symbol = st.selectbox("Choose a symbol", display_df["Symbol"].tolist())
selected_row = display_df[display_df["Symbol"] == selected_symbol].iloc[0]

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Price", selected_row["Price"])
m2.metric("Score", selected_row["Score"], help=score_explanation(selected_row["Score"]))
m3.metric("Decision", selected_row["Decision"], help=decision_explanation(selected_row["Decision"]))
m4.metric("RSI14", selected_row["RSI14"])
m5.metric("RVOL", selected_row["RVOL"])

st.write("**Last bar time:**", selected_row["Last Bar"])
st.write("**Reasons:**", selected_row["Reasons"] if selected_row["Reasons"] else "No strong signals right now.")

try:
    raw_chart = download_data(selected_symbol, period="10d", interval="5m")
    chart_df = add_indicators(raw_chart).copy()
    chart_df = chart_df.dropna(subset=["VWAP", "MA20", "MA50", "MA200"])

    fig = build_candlestick_chart(chart_df, selected_symbol)
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.warning(f"Could not load chart for {selected_symbol}: {e}")
