# =========================================================
#  REALISM TEST: Large Cap Only + Cash Filter
#  Universe: Nikkei 225 Major Stocks (No Small Caps)
#  Logic: If Momentum < 0 -> Cash (Wait)
# =========================================================

import sys
import pandas as pd
import numpy as np
import talib
import matplotlib.pyplot as plt
import yfinance as yf
from dateutil.relativedelta import relativedelta
import warnings

warnings.filterwarnings('ignore')

# --- 1. 設定: 超大型株ユニバース (当時から有名な企業のみ) ---
# トヨタ、ソニー、三菱UFJ、ファストリ、任天堂、日立など
LARGE_CAPS = [
    "7203.T", "6758.T", "8306.T", "9983.T", "7974.T", "6501.T", # トヨタ, ソニー, UFJ, ユニクロ, 任天堂, 日立
    "9984.T", "9432.T", "8035.T", "6098.T", "4502.T", "4503.T", # SBG, NTT, 東エレ, リクルート, 武田, アステラス
    "8058.T", "8001.T", "8031.T", "6301.T", "6954.T", "7741.T", # 三菱商, 伊藤忠, 三井物, コマツ, ファナック, HOYA
    "4063.T", "6902.T", "6981.T", "4452.T", "6273.T", "6594.T", # 信越, デンソー, 村田, 花王, SMC, 日本電産
    "7751.T", "8766.T", "8801.T", "9020.T", "9022.T", "2914.T"  # キヤノン, 東京海上, 三井不, JR東, JR東海, JT
]

START_DATE = "1995-01-01"
END_DATE = "2025-11-25"
INITIAL_CAPITAL = 3_000_000
COST_RATE = 0.001
REBALANCE_MONTHS = 6
TOP_N = 5
STOP_LOSS_PCT = 0.10

# --- 2. データ取得 ---
print("🇯🇵 大型株データを取得中...")
try:
    data_all = yf.download(LARGE_CAPS, start=START_DATE, end=END_DATE, group_by="ticker", auto_adjust=True, progress=False)
    data_all.index = data_all.index.tz_localize(None)
    
    # 日経平均
    nikkei = yf.download("^N225", start=START_DATE, end=END_DATE, interval="1d", auto_adjust=True, progress=False)['Close']
    nikkei.index = nikkei.index.tz_localize(None)
except: pass

# --- 3. 前処理 ---
processed = {}
for t in LARGE_CAPS:
    try:
        if isinstance(data_all.columns, pd.MultiIndex):
            if t in data_all.columns.levels[0]: df = data_all[t].copy()
            else: continue
        else: df = data_all.copy()
        
        df = df.dropna()
        if len(df) < 200: continue

        # 指標
        df["sma50"] = talib.SMA(df["Close"], 50)
        df["sma200"] = talib.SMA(df["Close"], 200)
        df["prev_high"] = df["High"].shift(1)
        df["breakout"] = (df["Close"] > df["prev_high"])
        
        # モメンタム
        df["mom_6m"] = df["Close"] / df["Close"].shift(120) - 1
        
        processed[t] = df.dropna()
    except: pass

# --- 4. シミュレーション ---
def run_realism_test(mode_name):
    print(f"\n=== {mode_name} (Cash Filtered) ===")
    
    current = pd.to_datetime(START_DATE) + relativedelta(years=1)
    end = pd.to_datetime(END_DATE)
    
    equity_curve = [INITIAL_CAPITAL]
    dates = [current]
    current_capital = INITIAL_CAPITAL
    positions = {}
    
    while current < end:
        next_rebalance = current + relativedelta(months=REBALANCE_MONTHS)
        
        # 1. 銘柄選定
        candidates = []
        for t, df in processed.items():
            try:
                idx = df.index.get_indexer([current], method='pad')[0]
                row = df.iloc[idx]
                # 条件: SMA200より上 (ここが重要！不況時は誰も条件を満たさない)
                if row["Close"] > row["sma200"]:
                    candidates.append({"Ticker": t, "Score": row["mom_6m"]})
            except: pass
            
        df_rank = pd.DataFrame(candidates).sort_values("Score", ascending=False)
        active_tickers = list(df_rank.head(TOP_N)["Ticker"])
        
        # ★現金フィルター: 候補がいない、またはモメンタムがマイナスなら現金
        valid_tickers = []
        for t in active_tickers:
            # モメンタムがプラスの銘柄だけ買う
            score = df_rank[df_rank['Ticker']==t]['Score'].values[0]
            if score > 0:
                valid_tickers.append(t)
        
        active_tickers = valid_tickers

        if not active_tickers:
            # ログ: 全額キャッシュ
            # print(f"📅 {current.date()} ⚠️ 冬の時代 (Cash 100%)")
            pass
        else:
            # print(f"📅 {current.date()} 選抜: {active_tickers}")
            pass
            
        # 2. 運用
        period_days = pd.date_range(current, next_rebalance, freq="B")
        
        # 資金配分 (現金も含めたポートフォリオ)
        # 5枠あるうち、選抜が2つなら、残り3枠(60%)は現金で持つ
        allocation_per_slot = current_capital / TOP_N
        
        # --- Lazy Hold (Cash Filtered) ---
        if mode_name == "Real Lazy":
            for t in active_tickers:
                try:
                    idx = processed[t].index.get_indexer([current], method='pad')[0]
                    entry_price = processed[t].iloc[idx]["Close"]
                    
                    qty = allocation_per_slot / entry_price
                    cost = qty * entry_price * COST_RATE
                    
                    if current_capital >= qty * entry_price + cost:
                        current_capital -= (qty * entry_price + cost)
                        positions[t] = {'qty': qty, 'entry': entry_price}
                except: pass

        # 日次ループ
        for d in period_days:
            if d > end: break
            
            # AI Sniper (Stop Loss)
            if mode_name == "Real Sniper":
                # Exit
                remove_list = []
                for t, pos in positions.items():
                    if d not in processed[t].index: continue
                    row = processed[t].loc[d]
                    # Exit: SMA50割れ or -10%
                    if row['Close'] < row['sma50'] or row['Low'] <= pos['entry'] * (1-STOP_LOSS_PCT):
                        exit_p = row['Close'] if row['Close'] < row['sma50'] else pos['entry'] * (1-STOP_LOSS_PCT)
                        current_capital += exit_p * pos['qty'] * (1 - COST_RATE)
                        remove_list.append(t)
                for t in remove_list: del positions[t]
                
                # Entry
                for t in active_tickers:
                    if t in positions: continue
                    if d not in processed[t].index: continue
                    row = processed[t].loc[d]
                    if row['breakout']:
                        cost = allocation_per_slot * (1 + COST_RATE)
                        if current_capital >= cost:
                            qty = allocation_per_slot / row['Close']
                            current_capital -= cost
                            positions[t] = {'qty': qty, 'entry': row['Close']}

            # 資産集計
            total_val = current_capital
            for t, pos in positions.items():
                if d in processed[t].index:
                    price = processed[t].loc[d]["Close"]
                    total_val += price * pos['qty']
                else:
                    total_val += pos['entry'] * pos['qty']
            
            equity_curve.append(total_val)
            dates.append(d)
            
        # 期末リバランス
        for t, pos in positions.items():
            try:
                idx = processed[t].index.get_indexer([period_days[-1]], method='pad')[0]
                price = processed[t].iloc[idx]["Close"]
                current_capital += price * pos['qty'] * (1 - COST_RATE)
            except:
                current_capital += pos['entry'] * pos['qty']
        positions = {}
        current = next_rebalance

    return equity_curve, dates

# --- 5. 実行 ---
eq_real_ai, dates_real_ai = run_realism_test("Real Sniper")
eq_real_hold, dates_real_hold = run_realism_test("Real Lazy")

# 評価
def get_res(eq):
    s = pd.Series(eq)
    ret = (s.iloc[-1] - INITIAL_CAPITAL)/INITIAL_CAPITAL*100
    dd = (s/s.cummax()-1).min()*100
    return ret, dd

ret_ai, dd_ai = get_res(eq_real_ai)
ret_hold, dd_hold = get_res(eq_real_hold)

print("\n" + "="*50)
print(f"【大型株限定・現金フィルター付き (1995-2025)】")
print(f"🤖 Real Sniper: {int(eq_real_ai[-1]):,}円 (Ret {ret_ai:+.0f}% / DD {dd_ai:.1f}%)")
print(f"💪 Real Hold  : {int(eq_real_hold[-1]):,}円 (Ret {ret_hold:+.0f}% / DD {dd_hold:.1f}%)")
print("="*50)

plt.figure(figsize=(12, 6))
plt.plot(dates_real_ai, eq_real_ai, label='Real Sniper (Defensive)', color='red')
plt.plot(dates_real_hold, eq_real_hold, label='Real Hold (Aggressive)', color='green')
nikkei_plot = (nikkei / nikkei.asof(dates_real_ai[0])) * INITIAL_CAPITAL
plt.plot(dates_real_ai, nikkei_plot.reindex(dates_real_ai, method='ffill'), label='Nikkei 225', color='black', linestyle='--')
plt.yscale('log')
plt.title("Realistic Backtest: Large Caps Only + Cash Filter")
plt.grid(True, which="both", ls="--")
plt.legend()
plt.show()