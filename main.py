#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scalp_engine.py
================
超短线剥头皮 (Scalping) 交易决策辅助计算引擎。

设计原则
--------
1. 空间计算全部基于用户传入的 Trading212 实时【现价】与【点差】,
   使用"相对位移 Delta"进行运算,避免不同数据源之间的绝对报价偏差。
2. 外部行情源 (yfinance) 仅用于提取【波动率】与【微观漂移率】这两个
   统计量,不作为价格基准 —— 这是本脚本最重要的设计约束。
3. 概率模型采用带漂移的【算术维纳过程】(Arithmetic Brownian Motion
   with drift),而非几何布朗运动。原因:用户的目标位是绝对价差
   (Delta),而不是收益率,算术 BM 与点差的线性单位天然一致。

免责声明
--------
本脚本输出的"触及概率"基于简化的随机过程假设(恒定漂移率、恒定
波动率、正态扩散),不构成对未来价格路径的保证。真实市场存在跳空、
流动性冲击、订单簿失衡等模型未覆盖的风险,仅供概率参考,不构成
投资建议。

依赖
----
    pip install yfinance numpy pandas scipy

命令行用法
----------
    python scalp_engine.py \
        --symbol Silver \
        --current 66.00 \
        --target 66.50 \
        --timeframe 30 \
        --spread 0.02 \
        --interval 1m

    --symbol 支持俗称/自然语言输入(Silver, US100, Oil, NVDA 等),
    内部由 resolve_ticker() 通过 SYMBOL_MAP 自动映射为 yfinance ticker
    (如 SI=F, ^NDX, CL=F);未命中映射表时原样透传,假定输入已是合法
    ticker(如 AAPL、SI=F 本身)。

也可以用 --mock 跳过网络请求,用合成数据自测(离线开发/CI 用)。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ #
# 常量
# ------------------------------------------------------------------ #
MIN_NET_RR = 1.2          # 净盈亏比硬门槛
PROB_FALLBACK_THRESHOLD = 0.40   # 低于此概率触发 65% 置信度回退
FALLBACK_CONFIDENCE = 0.65       # 回退止盈的目标置信度
SL_ATR_MULTIPLIER = 1.2   # SL = 1.2 × ATR(1m)
DEFAULT_LOOKBACK_BARS = 30  # 漂移率/波动率估计所用的最近 K 线数

# ------------------------------------------------------------------ #
# Symbol 映射表:自然语言/俗称 -> yfinance 可识别 Ticker
# ------------------------------------------------------------------ #
# 说明:
#   - key 一律小写、去除空格,匹配时对用户输入做同样的归一化处理。
#   - value 为 yfinance ticker。同一品种若有多个常见叫法,都在此列出。
#   - 这张表只覆盖 Trading212 常见 CFD 品种的一部分,不是穷举;
#     真要接全品种池,需要你按产品线持续补充。
#   - 未命中表内任何 key 时,resolve_ticker 会原样返回用户输入
#     (假定用户直接给的就是合法 ticker,如 NVDA、AAPL)。
SYMBOL_MAP: dict[str, str] = {
    # --- 贵金属 ---
    "silver": "SI=F",
    "xag": "SI=F",
    "xagusd": "SI=F",
    "白银": "SI=F",
    "gold": "GC=F",
    "xau": "GC=F",
    "xauusd": "GC=F",
    "黄金": "GC=F",

    # --- 能源 ---
    "oil": "CL=F",
    "crude oil": "CL=F",
    "crudeoil": "CL=F",
    "wti": "CL=F",
    "usoil": "CL=F",
    "原油": "CL=F",
    "brent": "BZ=F",
    "布伦特": "BZ=F",
    "natural gas": "NG=F",
    "naturalgas": "NG=F",
    "天然气": "NG=F",

    # --- 股指 ---
    "us100": "^NDX",
    "nas100": "^NDX",
    "nasdaq": "^NDX",
    "nasdaq100": "^NDX",
    "纳指": "^NDX",
    "us500": "^GSPC",
    "spx500": "^GSPC",
    "sp500": "^GSPC",
    "标普": "^GSPC",
    "标普500": "^GSPC",
    "us30": "^DJI",
    "dow": "^DJI",
    "dowjones": "^DJI",
    "道指": "^DJI",
    "uk100": "^FTSE",
    "ftse100": "^FTSE",
    "germany40": "^GDAXI",
    "dax40": "^GDAXI",
    "dax": "^GDAXI",
    "德指": "^GDAXI",
    "japan225": "^N225",
    "nikkei225": "^N225",
    "日经": "^N225",

    # --- 外汇 ---
    "eurusd": "EURUSD=X",
    "欧美": "EURUSD=X",
    "gbpusd": "GBPUSD=X",
    "镑美": "GBPUSD=X",
    "usdjpy": "USDJPY=X",
    "美日": "USDJPY=X",
    "audusd": "AUDUSD=X",
    "usdcad": "USDCAD=X",
    "usdchf": "USDCHF=X",

    # --- 加密货币 ---
    "btc": "BTC-USD",
    "bitcoin": "BTC-USD",
    "比特币": "BTC-USD",
    "eth": "ETH-USD",
    "ethereum": "ETH-USD",
    "以太坊": "ETH-USD",

    # --- 常见个股(直接就是合法 ticker,列在此处仅为显式兜底/防歧义) ---
    "nvda": "NVDA",
    "nvidia": "NVDA",
    "英伟达": "NVDA",
    "aapl": "AAPL",
    "apple": "AAPL",
    "苹果": "AAPL",
    "tsla": "TSLA",
    "tesla": "TSLA",
    "特斯拉": "TSLA",
    "msft": "MSFT",
    "microsoft": "MSFT",
    "微软": "MSFT",
}


def resolve_ticker(symbol: str) -> str:
    """
    将用户输入的自然语言/俗称标的名,解析为 yfinance 可识别的 ticker。

    匹配策略:
      1. 归一化输入(去首尾空格、转小写),精确匹配 SYMBOL_MAP。
      2. 未命中时,原样返回归一化前的输入(去除首尾空格),视为用户
         已直接提供合法 ticker(如 "AAPL"、"SI=F"),不做任何猜测性
         改写——宁可让 yfinance 自己报错,也不做静默的错误映射。

    例:
        resolve_ticker("Silver")  -> "SI=F"
        resolve_ticker("US100")   -> "^NDX"
        resolve_ticker("Oil")     -> "CL=F"
        resolve_ticker("NVDA")    -> "NVDA"   (未命中表,原样返回)
        resolve_ticker(" nvda ")  -> "NVDA"   (未命中表,但去除了空格)
    """
    if not symbol or not symbol.strip():
        raise ValueError("symbol 不能为空。")

    raw = symbol.strip()
    key = raw.lower()

    if key in SYMBOL_MAP:
        return SYMBOL_MAP[key]

    # 未命中映射表:假定输入本身已是合法 ticker,原样返回(仅去空格)
    return raw


# ------------------------------------------------------------------ #
# 数据结构
# ------------------------------------------------------------------ #
@dataclass
class MarketStats:
    """从外部数据源提取的微观结构统计量(全部为"每根1分钟K线"单位)。"""
    atr_1m: float              # ATR(14),价格单位,用于 SL 定距
    sigma_price_per_min: float  # 波动率,价格单位/分钟^0.5 (扩散系数)
    mu_price_per_min: float     # 漂移率,价格单位/分钟
    parkinson_vol_return: float  # 帕金森波动率(收益率口径,未换算价格)
    ema_slope_per_bar: float    # EMA(8) 斜率,价格单位/分钟
    log_return_drift_per_bar: float  # 对数收益均值漂移(收益率口径)
    bars_used: int


@dataclass
class EngineResult:
    direction: str
    hit_probability: float
    current_price: float
    spread: float
    tp_price: float
    sl_price: float
    net_risk_reward: float
    tradeable: bool
    reason: str
    resolved_symbol: str = ""  # 实际用于拉取行情的 yfinance ticker(供LLM前端核对映射是否正确)

    def to_json(self) -> str:
        d = asdict(self)
        # 统一四舍五入,避免浮点噪声污染输出
        for k in ("hit_probability",):
            d[k] = round(d[k], 4)
        for k in ("current_price", "spread", "tp_price", "sl_price", "net_risk_reward"):
            if d[k] is not None:
                d[k] = round(d[k], 6)
        return json.dumps(d, ensure_ascii=False)


# ------------------------------------------------------------------ #
# 第一部分:数据获取
# ------------------------------------------------------------------ #
def fetch_intraday_bars(symbol: str, interval: str = "1m", period: str = "1d") -> pd.DataFrame:
    """
    通过 yfinance 拉取高频 K 线。

    注意: yfinance 对 1m 数据通常只保留最近 7 天,对流动性较差的品种
    (部分 CFD 对应的期货/现货代码)可能返回空表或延迟数据。这里只用
    它来估计"波动率结构"和"动量斜率",不用于绝对价格,因此这类误差
    对最终结果的影响是二阶的。
    """
    try:
        import yfinance as yf
    except ImportError as e:
        raise RuntimeError(
            "未安装 yfinance,请先执行: pip install yfinance --break-system-packages"
        ) from e

    df = yf.download(
        tickers=symbol,
        interval=interval,
        period=period,
        progress=False,
        auto_adjust=False,
    )

    if df is None or df.empty:
        # 1m 数据常受限于 7 天窗口,尝试退化到更长 interval 兜底
        if interval == "1m":
            df = yf.download(
                tickers=symbol, interval="5m", period="5d",
                progress=False, auto_adjust=False,
            )
    if df is None or df.empty:
        raise RuntimeError(f"未能获取 {symbol} 的高频行情数据,数据源返回为空。")

    # yfinance 新版本可能返回 MultiIndex 列(多标的),这里做单标的展平
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns=str.title)  # Open/High/Low/Close/Volume
    required = {"Open", "High", "Low", "Close"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"行情数据列缺失,拿到的列为: {list(df.columns)}")

    return df.dropna(subset=["Open", "High", "Low", "Close"])


def make_mock_bars(n: int = 60, start_price: float = 66.0, seed: int = 7) -> pd.DataFrame:
    """离线自测用的合成 1 分钟 K 线(几何随机游走 + 轻微漂移)。"""
    rng = np.random.default_rng(seed)
    mu, sigma = 0.00005, 0.0009  # 每根K线的对数收益漂移/波动(合成参数)
    log_rets = rng.normal(mu, sigma, n)
    close = start_price * np.exp(np.cumsum(log_rets))
    open_ = np.roll(close, 1)
    open_[0] = start_price
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.0006, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.0006, n))
    idx = pd.date_range(end=pd.Timestamp.utcnow(), periods=n, freq="1min")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close}, index=idx
    )


# ------------------------------------------------------------------ #
# 第二部分:波动率 / 漂移率计算
# ------------------------------------------------------------------ #
def calc_atr(df: pd.DataFrame, period: int = 14) -> float:
    """标准 ATR(N),Wilder 平滑,返回价格单位的最新值。"""
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    val = atr.iloc[-1]
    if pd.isna(val) or val <= 0:
        # 数据不足 period 根时退化为简单 TR 均值
        val = tr.dropna().tail(period).mean()
    return float(val)


def calc_parkinson_volatility(df: pd.DataFrame, period: int = 14) -> float:
    """
    帕金森波动率估计量(收益率口径,无量纲):

        sigma_P = sqrt( 1 / (4 * N * ln2) * sum( ln(H_i / L_i)^2 ) )

    比传统 close-to-close 标准差更高效地利用了高低点信息,适合高频
    K 线的日内波动估计。
    """
    window = df.tail(period)
    hl_ratio = np.log(window["High"] / window["Low"])
    n = len(hl_ratio)
    if n == 0:
        return 0.0
    sigma = math.sqrt((hl_ratio ** 2).sum() / (4 * n * math.log(2)))
    return float(sigma)


def calc_drift_and_diffusion(
    df: pd.DataFrame, lookback: int = DEFAULT_LOOKBACK_BARS
) -> MarketStats:
    """
    综合计算漂移率 mu 与扩散系数 sigma,统一换算为"价格单位 / 分钟"。

    漂移率 mu (价格单位/bar):
        - EMA(8)/EMA(21) 斜率反映中期动量方向
        - 最近 N 根 K 线对数收益率均值反映短期动量强度
        - 二者加权平均,权重各 0.5

    扩散系数 sigma (价格单位/sqrt(分钟)):
        - 帕金森波动率(收益率口径)× 当前价格,换算为价格单位
    """
    if len(df) < 5:
        raise RuntimeError("K 线数量过少,无法估计漂移率/波动率(至少需要 5 根)。")

    close = df["Close"]
    last_price = float(close.iloc[-1])
    n = min(lookback, len(df))
    window = df.tail(n)

    # --- EMA 斜率(价格单位/bar) ---
    ema8 = close.ewm(span=8, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    slope_span = min(5, len(ema8) - 1)
    if slope_span < 1:
        ema_slope = 0.0
    else:
        ema_slope = float((ema8.iloc[-1] - ema8.iloc[-1 - slope_span]) / slope_span)
    # EMA8 相对 EMA21 的位置作为动量方向一致性修正(避免逆势EMA8斜率误判)
    trend_bias = np.sign(ema8.iloc[-1] - ema21.iloc[-1])
    if np.sign(ema_slope) != 0 and trend_bias != 0 and np.sign(ema_slope) != trend_bias:
        ema_slope *= 0.5  # 斜率与趋势方向背离时,打折信任

    # --- 对数收益均值漂移(收益率口径 -> 价格单位/bar) ---
    log_rets = np.log(window["Close"] / window["Close"].shift(1)).dropna()
    log_return_mean = float(log_rets.mean()) if len(log_rets) else 0.0
    log_drift_price = log_return_mean * last_price

    mu_price = 0.5 * ema_slope + 0.5 * log_drift_price

    # --- 帕金森波动率 -> 价格单位扩散系数(每分钟) ---
    parkinson = calc_parkinson_volatility(df, period=n)
    sigma_price = parkinson * last_price
    # 若帕金森估计异常为 0(数据点过少或高低完全相等),退化为收益率标准差
    if sigma_price <= 0:
        sigma_price = float(log_rets.std() * last_price) if len(log_rets) > 1 else 1e-6
    sigma_price = max(sigma_price, 1e-8)  # 防止除零

    atr_1m = calc_atr(df, period=14)

    return MarketStats(
        atr_1m=atr_1m,
        sigma_price_per_min=sigma_price,
        mu_price_per_min=mu_price,
        parkinson_vol_return=parkinson,
        ema_slope_per_bar=ema_slope,
        log_return_drift_per_bar=log_return_mean,
        bars_used=n,
    )


# ------------------------------------------------------------------ #
# 第三部分:首达时间 (First Passage Time) 概率模型
# ------------------------------------------------------------------ #
def fpt_hit_probability(mu: float, sigma: float, T: float, barrier: float) -> float:
    """
    带漂移的算术维纳过程 X(t) = mu*t + sigma*W(t), X(0)=0。

    计算 P( 存在 s in [0, T], 使得 X(s) 越过 barrier )。

    标准闭式解(barrier = a > 0,即向上突破):
        P = Phi( (mu*T - a) / (sigma*sqrt(T)) )
            + exp(2*mu*a / sigma^2) * Phi( (-mu*T - a) / (sigma*sqrt(T)) )

    对于 a < 0(向下突破),通过反射 X -> -X, mu -> -mu, a -> -a 转化为
    上述同一形式求解,数学上完全对称,故本函数内部统一处理两种情形。

    参数
    ----
    mu      : 漂移率,价格单位 / 分钟
    sigma   : 扩散系数,价格单位 / sqrt(分钟)
    T       : 时间窗口,分钟
    barrier : 目标位相对现价的有向位移 (target_price - current_price)

    返回
    ----
    float: 落在 [0, 1] 的触及概率
    """
    if T <= 0:
        return 0.0
    if barrier == 0:
        return 1.0
    if sigma <= 0:
        # 无波动率时,是否命中完全由漂移方向的确定性路径决定
        reached = (mu * T >= barrier) if barrier > 0 else (mu * T <= barrier)
        return 1.0 if reached else 0.0

    # 统一反射到 a>0, m 为等效漂移 的形式
    if barrier > 0:
        m, a = mu, barrier
    else:
        m, a = -mu, -barrier

    sqrtT = math.sqrt(T)
    d1 = (m * T - a) / (sigma * sqrtT)
    d2 = (-m * T - a) / (sigma * sqrtT)

    # exp(2*m*a/sigma^2) 在 m,a 很大时可能数值溢出,做对数域裁剪保护
    exponent = 2 * m * a / (sigma ** 2)
    exponent = min(exponent, 700.0)  # exp(700) 已接近 float64 上限,防止 overflow
    term2 = math.exp(exponent) * norm.cdf(d2)

    p = norm.cdf(d1) + term2
    return float(min(max(p, 0.0), 1.0))


def solve_barrier_for_confidence(
    mu: float, sigma: float, T: float, target_prob: float, direction_sign: int,
    search_upper_mult: float = 60.0,
) -> Optional[float]:
    """
    反解:给定目标置信度 target_prob(如 0.65),在 direction_sign 指示的
    方向上(+1 = 向上 / LONG,-1 = 向下 / SHORT),求解使
    fpt_hit_probability(mu, sigma, T, barrier) == target_prob 的 barrier。

    fpt_hit_probability 关于 |barrier| 是单调递减的(离现价越远,越难在
    T 分钟内触及),因此在 (epsilon, upper_bound) 上用 brentq 二分求根。
    """
    if sigma <= 0 or T <= 0:
        return None

    def f(mag: float) -> float:
        return fpt_hit_probability(mu, sigma, T, direction_sign * mag) - target_prob

    eps = 1e-9
    upper = search_upper_mult * sigma * math.sqrt(T) + abs(mu) * T + 1.0

    f_low = f(eps)
    f_high = f(upper)

    # 若上界仍未跌破目标概率(说明漂移极强,继续外推上界)
    tries = 0
    while f_low * f_high > 0 and tries < 6:
        upper *= 2
        f_high = f(upper)
        tries += 1

    if f_low * f_high > 0:
        # 无法在合理范围内找到根(极端漂移下概率恒高于/低于目标)
        return None

    mag_root = brentq(f, eps, upper, xtol=1e-10, maxiter=200)
    return direction_sign * mag_root


# ------------------------------------------------------------------ #
# 第四部分:点差过滤 + 动态 TP/SL + 决策整合
# ------------------------------------------------------------------ #
def run_engine(
    symbol: str,
    current_price: float,
    target_price: float,
    timeframe_min: float,
    spread: float,
    interval: str = "1m",
    period: str = "1d",
    lookback_bars: int = DEFAULT_LOOKBACK_BARS,
    mock: bool = False,
) -> EngineResult:
    """主计算入口:整合波动率/漂移率估计、FPT 概率、点差过滤、动态 TP/SL。"""

    if current_price <= 0 or spread < 0 or timeframe_min <= 0:
        raise ValueError("current_price 必须为正,spread 不能为负,timeframe_min 必须为正。")

    # ---------- 0. Symbol 解析:自然语言/俗称 -> yfinance ticker ----------
    resolved_symbol = resolve_ticker(symbol)

    # ---------- 1. 获取行情 & 估计统计量 ----------
    df = make_mock_bars(start_price=current_price) if mock else fetch_intraday_bars(
        resolved_symbol, interval=interval, period=period
    )
    stats = calc_drift_and_diffusion(df, lookback=lookback_bars)

    mu = stats.mu_price_per_min
    sigma = stats.sigma_price_per_min
    atr_1m = stats.atr_1m

    # ---------- 2. 目标位方向与原始 FPT 概率 ----------
    delta_target = target_price - current_price
    if delta_target > 0:
        direction = "LONG"
        sign = 1
    elif delta_target < 0:
        direction = "SHORT"
        sign = -1
    else:
        return EngineResult(
            direction="NO_TRADE", hit_probability=0.0, current_price=current_price,
            spread=spread, tp_price=current_price, sl_price=current_price,
            net_risk_reward=0.0, tradeable=False,
            reason="目标价与现价相同,不存在有效交易空间。",
            resolved_symbol=resolved_symbol,
        )

    p_target = fpt_hit_probability(mu, sigma, timeframe_min, delta_target)

    # ---------- 3. 点差硬过滤:净目标空间必须为正 ----------
    net_target_space = abs(delta_target) - spread
    sl_distance = SL_ATR_MULTIPLIER * atr_1m
    if sl_distance <= 0:
        sl_distance = max(spread * 2, 1e-6)  # ATR 异常时兜底,避免除零

    sl_price = current_price - sign * sl_distance

    if net_target_space <= 0:
        return EngineResult(
            direction="NO_TRADE",
            hit_probability=round(p_target, 4),
            current_price=current_price,
            spread=spread,
            tp_price=target_price,
            sl_price=round(sl_price, 6),
            net_risk_reward=0.0,
            tradeable=False,
            reason=(
                f"点差侵蚀严重:目标空间 {abs(delta_target):.5f} 不足以覆盖点差 "
                f"{spread:.5f},净空间 <= 0,判定为不可交易。"
            ),
            resolved_symbol=resolved_symbol,
        )

    # ---------- 4. 净盈亏比校验(先用原始目标位估算) ----------
    net_rr = net_target_space / sl_distance

    if net_rr < MIN_NET_RR:
        return EngineResult(
            direction="NO_TRADE",
            hit_probability=round(p_target, 4),
            current_price=current_price,
            spread=spread,
            tp_price=round(current_price + sign * net_target_space, 6),
            sl_price=round(sl_price, 6),
            net_risk_reward=round(net_rr, 4),
            tradeable=False,
            reason=(
                f"扣除点差后净盈亏比 {net_rr:.2f} < {MIN_NET_RR},"
                f"止损距离({sl_distance:.5f})相对目标空间过大,判定为不可交易。"
            ),
            resolved_symbol=resolved_symbol,
        )

    # ---------- 5. 概率门槛判断:低于 40% 则回退至 65% 置信度止盈位 ----------
    if p_target >= PROB_FALLBACK_THRESHOLD:
        # 原目标位可行:TP = 目标价扣除点差
        tp_price = current_price + sign * net_target_space
        hit_probability = p_target
        reason = (
            f"目标位触及概率 {p_target:.1%} 高于 {PROB_FALLBACK_THRESHOLD:.0%} 门槛,"
            f"净盈亏比 {net_rr:.2f},顺势{'做多' if direction == 'LONG' else '做空'}。"
        )
        tradeable = True
    else:
        # 概率不足,反推 65% 置信度下的合理止盈位
        b65 = solve_barrier_for_confidence(
            mu, sigma, timeframe_min, FALLBACK_CONFIDENCE, sign
        )
        if b65 is None:
            return EngineResult(
                direction="NO_TRADE",
                hit_probability=round(p_target, 4),
                current_price=current_price,
                spread=spread,
                tp_price=round(current_price + sign * net_target_space, 6),
                sl_price=round(sl_price, 6),
                net_risk_reward=round(net_rr, 4),
                tradeable=False,
                reason=(
                    f"目标位概率仅 {p_target:.1%},且当前波动率/漂移率结构下无法"
                    f"反解出 {FALLBACK_CONFIDENCE:.0%} 置信度的合理止盈位,判定为不可交易。"
                ),
                resolved_symbol=resolved_symbol,
            )

        fallback_tp = current_price + b65
        fallback_net_space = abs(b65) - spread

        if fallback_net_space <= 0:
            return EngineResult(
                direction="NO_TRADE",
                hit_probability=round(p_target, 4),
                current_price=current_price,
                spread=spread,
                tp_price=round(fallback_tp, 6),
                sl_price=round(sl_price, 6),
                net_risk_reward=0.0,
                tradeable=False,
                reason=(
                    f"原目标概率仅 {p_target:.1%},{FALLBACK_CONFIDENCE:.0%} 置信度回退位"
                    f"({fallback_tp:.5f})距现价过近,扣除点差后空间 <= 0,不可交易。"
                ),
                resolved_symbol=resolved_symbol,
            )

        fallback_rr = fallback_net_space / sl_distance
        if fallback_rr < MIN_NET_RR:
            return EngineResult(
                direction="NO_TRADE",
                hit_probability=round(p_target, 4),
                current_price=current_price,
                spread=spread,
                tp_price=round(fallback_tp, 6),
                sl_price=round(sl_price, 6),
                net_risk_reward=round(fallback_rr, 4),
                tradeable=False,
                reason=(
                    f"原目标概率仅 {p_target:.1%},{FALLBACK_CONFIDENCE:.0%} 置信度回退位"
                    f"净盈亏比 {fallback_rr:.2f} 仍 < {MIN_NET_RR},不可交易。"
                ),
                resolved_symbol=resolved_symbol,
            )

        tp_price = current_price + sign * fallback_net_space  # 扣点差后的实际可用TP
        hit_probability = FALLBACK_CONFIDENCE
        net_rr = fallback_rr
        reason = (
            f"原目标位({target_price})触及概率仅 {p_target:.1%},低于 "
            f"{PROB_FALLBACK_THRESHOLD:.0%} 门槛;已回退至 {FALLBACK_CONFIDENCE:.0%} "
            f"置信度精细止盈位 {tp_price:.5f},净盈亏比 {net_rr:.2f}。"
        )
        tradeable = True

    return EngineResult(
        direction=direction,
        hit_probability=round(hit_probability, 4),
        current_price=current_price,
        spread=spread,
        tp_price=round(tp_price, 6),
        sl_price=round(sl_price, 6),
        net_risk_reward=round(net_rr, 4),
        tradeable=tradeable,
        reason=reason,
        resolved_symbol=resolved_symbol,
    )


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="超短线剥头皮交易决策引擎 (FPT 概率 + 点差硬过滤 + 动态TP/SL)"
    )
    p.add_argument(
        "--symbol", required=True,
        help="标的代码或俗称,如 Silver / US100 / Oil / NVDA / SI=F,会经 resolve_ticker 自动映射",
    )
    p.add_argument("--current", type=float, required=True, help="Trading212 实时现价")
    p.add_argument("--target", type=float, required=True, help="用户目标价格")
    p.add_argument("--timeframe", type=float, required=True, help="目标时间窗口(分钟)")
    p.add_argument("--spread", type=float, required=True, help="Trading212 实时点差")
    p.add_argument("--interval", default="1m", help="K线周期,默认1m")
    p.add_argument("--period", default="1d", help="K线拉取范围,默认1d")
    p.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK_BARS, help="漂移/波动率估计所用K线根数")
    p.add_argument("--mock", action="store_true", help="离线模式:用合成数据代替真实行情(用于自测)")
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        result = run_engine(
            symbol=args.symbol,
            current_price=args.current,
            target_price=args.target,
            timeframe_min=args.timeframe,
            spread=args.spread,
            interval=args.interval,
            period=args.period,
            lookback_bars=args.lookback,
            mock=args.mock,
        )
        print(result.to_json())
        return 0
    except Exception as e:
        # 出错也输出标准 JSON 结构,便于 LLM 前端统一解析
        try:
            resolved = resolve_ticker(args.symbol)
        except Exception:
            resolved = args.symbol
        err_result = {
            "direction": "NO_TRADE",
            "hit_probability": 0.0,
            "current_price": args.current,
            "spread": args.spread,
            "tp_price": None,
            "sl_price": None,
            "net_risk_reward": 0.0,
            "tradeable": False,
            "reason": f"引擎异常: {str(e)}",
            "resolved_symbol": resolved,
        }
        print(json.dumps(err_result, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
