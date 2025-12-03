# =========================================================
#  US SECTOR ROTATION: 25-Year History Test (1999-2024)
#  Logic: Dual Momentum (Sector Select + Regime Filter)
#  No Survivorship Bias (Using ETFs)
# =========================================================

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import talib
from dateutil.relativedelta import relativedelta
import warnings

warnings.filterwarnings('ignore')

# --- 1. 設定 ---
# SPDR Select Sector ETFs (1998年設定)
SECTORS = [
    "XLK", # Technology
    "XLF", # Financials
    "XLV", # Health Care
    "XLE", # Energy
    "XLY", # Consumer Discretionary
    "XLI", # Industrials
    "XLP", # Consumer Staples
    "XLU", # Utilities
    "XLB"  # Materials
]
# 安全資産
# TLT(2002~), GLD(2004~) はデータがない期間があるので注意
SAFE_ASSETS = ["TLT", "GLD", "SHY"] 
BENCHMARK = "^GSPC" # S&P 500

ALL_TICKERS = SECTORS + SAFE_ASSETS + [BENCHMARK]

START_DATE = "1999-01-01"
END_DATE = "2025-01-01"
INITIAL_CAPITAL = 10000
REBALANCE_MONTHS = 1 # 毎月チェック推奨
COST_RATE = 0.001

# --- 2. データ取得 ---
print("🇺🇸 25年分のETFデータを取得中...")
try:
    data_all = yf.download(ALL_TICKERS, start=START_DATE, end=END_DATE, group_by="ticker", auto_adjust=True, progress=False)
    data_all.index = data_all.index.tz_localize(None)
except: sys.exit()

# 前処理
processed = {}
for t in ALL_TICKERS:
    try:
        if isinstance(data_all.columns, pd.MultiIndex):
            if t in data_all.columns.levels[0]: df = data_all[t].copy()
            else: continue
        else: df = data_all.copy()
        
        df = df.dropna()
        if len(df) < 100: continue

        # 指標
        df["mom_6m"] = df["Close"] / df["Close"].shift(120) - 1
        df["sma200"] = df["Close"].rolling(200).mean()
        
        processed[t] = df.dropna()
    except: pass

# --- 3. シミュレーション ---
print("\n=== 🔄 セクター・ローテーション開始 ===")

current = pd.to_datetime(START_DATE) + relativedelta(years=1)
end = pd.to_datetime(END_DATE)

equity_curve = [INITIAL_CAPITAL]
dates = [current]
current_capital = INITIAL_CAPITAL
allocation_history = []

while current < end:
    next_rebalance = current + relativedelta(months=REBALANCE_MONTHS)
    
    # 1. 市場環境判定 (S&P500)
    is_bull = False
    if BENCHMARK in processed:
        try:
            idx = processed[BENCHMARK].index.get_indexer([current], method='pad')[0]
            row = processed[BENCHMARK].iloc[idx]
            if row["Close"] > row["sma200"]:
                is_bull = True
        except: pass
        
    # 2. 対象決定
    target_assets = []
    
    if is_bull:
        # 強気相場: セクター上位3つ
        scores = []
        for t in SECTORS:
            if t in processed:
                try:
                    idx = processed[t].index.get_indexer([current], method='pad')[0]
                    # データが古い(上場前)場合は除外
                    if (current - processed[t].index[idx]).days < 10:
                        scores.append((t, processed[t].iloc[idx]["mom_6m"]))
                except: pass
        
        scores.sort(key=lambda x: x[1], reverse=True)
        # モメンタムがプラスのものだけ
        target_assets = [x[0] for x in scores[:3] if x[1] > 0]
        
        # もしプラスのセクターがなければ守りへ
        if not target_assets: is_bull = False
            
    if not is_bull:
        # 弱気相場: 安全資産の中でモメンタム最強のもの
        safe_scores = []
        for t in SAFE_ASSETS:
            if t in processed:
                try:
                    idx = processed[t].index.get_indexer([current], method='pad')[0]
                    if (current - processed[t].index[idx]).days < 10:
                        # 安全資産もSMA200超えなら買う、そうでなければ現金
                        row = processed[t].iloc[idx]
                        if row["Close"] > row["sma200"]:
                            safe_scores.append((t, row["mom_6m"]))
                except: pass
        
        safe_scores.sort(key=lambda x: x[1], reverse=True)
        if safe_scores:
            target_assets = [safe_scores[0][0]] # 最強の盾を1つ選ぶ
        else:
            target_assets = [] # 完全現金化

    # ログ用
    alloc = {'Date': current, 'Cash': 1.0 if not target_assets else 0}
    for t in SECTORS + SAFE_ASSETS:
        alloc[t] = 0
    if target_assets:
        w = 1.0 / len(target_assets)
        for t in target_assets: alloc[t] = w
    allocation_history.append(alloc)

    # 3. 期間実行
    period_days = pd.date_range(current, next_rebalance, freq="B")
    period_daily_ret = pd.Series(0.0, index=period_days)
    
    if target_assets:
        weight = 1.0 / len(target_assets)
        for t in target_assets:
            if t in processed:
                df_period = processed[t].loc[current:next_rebalance]
                if not df_period.empty:
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
            
    current = next_rebalance

# --- 5. 評価 ---
final_ret = (equity_curve[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
s = pd.Series(equity_curve)
mdd = (s / s.cummax() - 1).min() * 100

# S&P500
sp500_full = processed["^GSPC"]["Close"]
sp_norm = (sp500_full / sp500_full.asof(dates[0])) * INITIAL_CAPITAL
sp_aligned = sp_norm.reindex(dates, method='ffill')

print("\n" + "="*50)
print(f"【米国株セクター・ローテーション (1999-2024)】")
print(f"🚀 AI Strategy : ${int(equity_curve[-1]):,} ({final_ret:+.0f}%) | MaxDD: {mdd:.1f}%")
print(f"🇺🇸 S&P 500     : ${int(sp_aligned.iloc[-1]):,} ({(sp_aligned.iloc[-1]/INITIAL_CAPITAL-1)*100:+.0f}%)")
print("="*50)

# グラフ
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

ax1.plot(dates, equity_curve, label='Sector Rotation', color='blue', linewidth=2)
ax1.plot(dates, sp_aligned, label='S&P 500', color='red', linestyle='--', alpha=0.7)
ax1.set_yscale('log')
ax1.set_title("25-Year Performance: Escaping Bubbles & Crashes")
ax1.set_ylabel("Capital (USD)")
ax1.legend()
ax1.grid(True, which="both", ls="--")

# 配分推移
df_alloc = pd.DataFrame(allocation_history).set_index('Date').fillna(0)
# 主要なものだけ表示
plot_cols = ["XLK", "XLF", "XLE", "XLV", "TLT", "GLD", "Cash"]
# Cash列がない場合のケア
if 'Cash' not in df_alloc.columns: df_alloc['Cash'] = 0

ax2.stackplot(df_alloc.index, df_alloc[plot_cols].T, labels=plot_cols, alpha=0.8, cmap='tab10')
ax2.set_title("Asset Allocation (Risk On / Risk Off)")
ax2.set_ylabel("Weight")
ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()