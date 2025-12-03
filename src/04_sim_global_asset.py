# =========================================================
#  GLOBAL ASSET ROTATION (2005-2025)
#  Universe: US, Developed, Emerging, REIT, Gold, Bonds
#  Logic: Trend Follow (Momentum) + Regime Filter
# =========================================================

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from dateutil.relativedelta import relativedelta
import warnings

warnings.filterwarnings('ignore')

# --- 1. 設定 ---
# グローバル資産クラスETF
ASSETS = {
    "SPY": "US Stocks",
    "EFA": "Dev ex-US", # 先進国(欧州・日本)
    "EEM": "Emerging",  # 新興国
    "VNQ": "US REIT",   # 不動産
    "GLD": "Gold",      # 金
    "TLT": "US Bonds"   # 国債
}
TICKERS = list(ASSETS.keys())
BENCHMARK = "SPY" # 比較対象

START_DATE = "2005-01-01" # EEM, GLDなどが揃う時期
END_DATE = "2025-01-01"
INITIAL_CAPITAL = 10000
REBALANCE_MONTHS = 1
COST_RATE = 0.001

# --- 2. データ取得 ---
print("🌍 グローバル資産データを取得中...")
try:
    data_all = yf.download(TICKERS, start=START_DATE, end=END_DATE, group_by="ticker", auto_adjust=True, progress=False)
    data_all.index = data_all.index.tz_localize(None)
except: sys.exit()

# 前処理
processed = {}
for t in TICKERS:
    try:
        if isinstance(data_all.columns, pd.MultiIndex):
            if t in data_all.columns.levels[0]: df = data_all[t].copy()
            else: continue
        else: df = data_all.copy()
        
        df = df.dropna()
        df["mom_6m"] = df["Close"] / df["Close"].shift(120) - 1
        df["sma200"] = df["Close"].rolling(200).mean()
        
        processed[t] = df.dropna()
    except: pass

# --- 3. シミュレーション ---
print("\n=== 🌍 グローバル・ローテーション開始 ===")

current = pd.to_datetime(START_DATE) + relativedelta(years=1)
end = pd.to_datetime(END_DATE)

equity_curve = [INITIAL_CAPITAL]
dates = [current]
current_capital = INITIAL_CAPITAL
allocation_history = []

while current < end:
    next_rebalance = current + relativedelta(months=REBALANCE_MONTHS)
    
    # 1. 候補選定 (SMA200超えのみ)
    candidates = []
    for t in TICKERS:
        if t in processed:
            try:
                idx = processed[t].index.get_indexer([current], method='pad')[0]
                row = processed[t].iloc[idx]
                if row["Close"] > row["sma200"]:
                    candidates.append((t, row["mom_6m"]))
            except: pass
            
    # 2. ランキング上位2つを選択
    candidates.sort(key=lambda x: x[1], reverse=True)
    # モメンタムがプラスのものだけ
    active_assets = [x[0] for x in candidates[:2] if x[1] > 0]
    
    # 3. ログ記録
    alloc = {'Date': current, 'Cash': 1.0 if not active_assets else 0}
    for t in TICKERS: alloc[t] = 0
    if active_assets:
        w = 1.0 / len(active_assets)
        for t in active_assets: alloc[t] = w
    allocation_history.append(alloc)

    # 4. 期間実行
    period_days = pd.date_range(current, next_rebalance, freq="B")
    period_daily_ret = pd.Series(0.0, index=period_days)
    
    if active_assets:
        weight = 1.0 / len(active_assets)
        for t in active_assets:
            if t in processed:
                df_period = processed[t].loc[current:next_rebalance]
                pct = df_period['Close'].pct_change().fillna(0)
                common = pct.index.intersection(period_daily_ret.index)
                period_daily_ret.loc[common] += pct.loc[common] * weight
        
        if len(period_daily_ret) > 0:
            period_daily_ret.iloc[0] -= COST_RATE

    # 資産更新
    equity_change = (1 + period_daily_ret).cumprod()
    start_cap = current_capital
    for d, ret_factor in equity_change.items():
        if d > dates[-1]:
            equity_curve.append(start_cap * ret_factor)
            dates.append(d)
            current_capital = start_cap * ret_factor
            
    current_capital -= current_capital * COST_RATE
    current = next_rebalance

# --- 5. 結果 ---
final_ret = (equity_curve[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
s = pd.Series(equity_curve)
mdd = (s / s.cummax() - 1).min() * 100

# SPY比較
spy_df = processed["SPY"]["Close"]
spy_norm = (spy_df / spy_df.asof(dates[0])) * INITIAL_CAPITAL
spy_aligned = spy_norm.reindex(dates, method='ffill')
spy_ret = (spy_aligned.iloc[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

print("\n" + "="*50)
print(f"【グローバル分散投資 (2006-2025)】")
print(f"🌍 AI Global : ${int(equity_curve[-1]):,} ({final_ret:+.0f}%) | MaxDD: {mdd:.1f}%")
print(f"🇺🇸 S&P 500   : ${int(spy_aligned.iloc[-1]):,} ({spy_ret:+.0f}%)")
print("="*50)

# グラフ
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

ax1.plot(dates, equity_curve, label='Global Rotation', color='green', linewidth=2)
ax1.plot(dates, spy_aligned, label='US Stocks Only (SPY)', color='gray', linestyle='--')
ax1.set_yscale('log')
ax1.set_title("Global Multi-Asset Rotation")
ax1.set_ylabel("Capital (USD)")
ax1.legend()
ax1.grid(True, which="both", ls="--")

# 配分推移
df_alloc = pd.DataFrame(allocation_history).set_index('Date').fillna(0)
ax2.stackplot(df_alloc.index, df_alloc.T, labels=df_alloc.columns, alpha=0.8, cmap='tab10')
ax2.set_title("Asset Allocation")
ax2.set_ylabel("Weight")
ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()




# =========================================================
#  ALL ASSETS BATTLE: AI vs The World
#  Benchmark: SPY, EFA, EEM, VNQ, GLD, TLT
# =========================================================

import matplotlib.pyplot as plt
import pandas as pd

# --- 前提: 前回のコードで `equity_curve`, `dates`, `processed` があること ---
# processed には全ETFのデータが入っているはずです

print("📊 全資産のパフォーマンスを比較中...")

# 1. 全資産の正規化 (Initial Capitalに合わせる)
asset_curves = {}
start_date = dates[0]

for t in TICKERS:
    if t in processed:
        # 開始日の価格を取得
        try:
            start_price = processed[t].loc[processed[t].index >= start_date].iloc[0]['Close']
            # 正規化: (価格 / 開始価格) * 10000
            curve = (processed[t]['Close'] / start_price) * INITIAL_CAPITAL
            # 日付合わせ (リサンプリング等はせず、そのままプロット用に保存)
            asset_curves[t] = curve.loc[start_date:]
        except: pass

# 2. グラフ描画
plt.figure(figsize=(14, 8))

# ベンチマーク群 (点線・細め)
colors = {
    "SPY": "red",    # 米国株
    "EFA": "orange", # 先進国
    "EEM": "brown",  # 新興国
    "VNQ": "purple", # 不動産
    "GLD": "gold",   # 金
    "TLT": "green"   # 国債
}

for t, curve in asset_curves.items():
    c = colors.get(t, "gray")
    # SPYだけ少し目立たせる、他は薄く
    alpha = 0.8 if t == "SPY" else 0.4
    width = 1.5 if t == "SPY" else 1.0
    plt.plot(curve.index, curve, label=f"{t} ({ASSETS[t]})", color=c, linestyle='--', alpha=alpha, linewidth=width)

# AI Bot (実線・太め・青)
plt.plot(dates, equity_curve, label='🚀 AI Global Rotation', color='blue', linewidth=3, zorder=10)

plt.yscale('log')
plt.title("AI Rotation vs Global Assets (2005-2025)")
plt.ylabel("Capital (USD - Log Scale)")
plt.legend(loc='upper left', bbox_to_anchor=(1, 1)) # 凡例を外に出す
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.show()

# --- 3. 最終順位の表示 ---
final_values = {t: curve.iloc[-1] for t, curve in asset_curves.items()}
final_values['AI Bot'] = equity_curve[-1]

# ランキング作成
ranking = pd.Series(final_values).sort_values(ascending=False)
print("\n【🏁 最終成績ランキング (20年間)】")
for rank, (name, val) in enumerate(ranking.items(), 1):
    ret = (val - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    print(f"{rank}位: {name:<10} ${int(val):,} ({ret:+.0f}%)")