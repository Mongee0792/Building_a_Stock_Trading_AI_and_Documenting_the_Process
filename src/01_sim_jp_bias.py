import sys
import pandas as pd
import numpy as np
import talib
import matplotlib.pyplot as plt
import yfinance as yf
from dateutil.relativedelta import relativedelta
import warnings

warnings.filterwarnings('ignore')

# --- 1. 設定 ---
TICKERS = [
    "6920.T","9984.T","6146.T","8035.T","6857.T","6526.T",
    "7203.T","7011.T","5401.T","6301.T",
    "8306.T","8316.T","8591.T","8766.T",
    "8058.T","8001.T",
    "9432.T","9983.T","4452.T","2914.T","3382.T",
    "9020.T","4661.T","4502.T","4568.T",
    "7974.T","6098.T","4385.T","8801.T","6758.T"
]

# ★★★ 変更点: 1995年開始 (データ安定のため) ★★★
START_DATE = "1995-01-01" 
END_DATE = "2025-11-25"
INITIAL_CAPITAL = 3_000_000
COST_RATE = 0.001
REBALANCE_MONTHS = 6
TOP_N = 5
STOP_LOSS_PCT = 0.10 # AI用

# --- 2. データ取得 ---
print(f"🇯🇵 データ取得中 ({START_DATE}から)...")
try:
    data_all = yf.download(TICKERS, start=START_DATE, end=END_DATE, group_by="ticker", auto_adjust=True, progress=False)
    data_all.index = data_all.index.tz_localize(None)
    
    # 日経平均
    nikkei = yf.download("^N225", start=START_DATE, end=END_DATE, interval="1d", auto_adjust=True, progress=False)['Close']
    nikkei.index = nikkei.index.tz_localize(None)
except: pass

# --- 3. 前処理 ---
print("指標計算中...")
processed = {}
for t in TICKERS:
    try:
        if isinstance(data_all.columns, pd.MultiIndex):
            if t in data_all.columns.levels[0]: df = data_all[t].copy()
            else: continue
        else: df = data_all.copy()
        
        df = df.dropna()
        if len(df) < 200: continue

        # 指標
        df["sma20"] = talib.SMA(df["Close"], 20)
        df["sma50"] = talib.SMA(df["Close"], 50)
        df["sma200"] = talib.SMA(df["Close"], 200)
        df["atr"] = talib.ATR(df["High"], df["Low"], df["Close"], 14)
        df["atr_ratio"] = df["atr"] / df["Close"]
        df["mom_6m"] = df["Close"] / df["Close"].shift(120) - 1
        df["prev_high"] = df["High"].shift(1)
        df["breakout"] = (df["Close"] > df["prev_high"])
        
        processed[t] = df.dropna()
    except: pass

# --- 4. シミュレーション関数 ---
def run_strategy(mode_name):
    print(f"\n=== {mode_name} 開始 ===")
    
    # 開始点をデータが揃った時点(約1年後)に設定
    current = pd.to_datetime(START_DATE) + relativedelta(years=1)
    end = pd.to_datetime(END_DATE)
    
    equity_curve = [INITIAL_CAPITAL]
    dates = [current]
    current_capital = INITIAL_CAPITAL
    positions = {} 
    
    while current < end:
        next_rebalance = current + relativedelta(months=REBALANCE_MONTHS)
        
        # 1. 銘柄選定 (Macro)
        candidates = []
        for t, df in processed.items():
            try:
                idx = df.index.get_indexer([current], method='pad')[0]
                row = df.iloc[idx]
                if row["Close"] > row["sma200"] and row["atr_ratio"] > 0.015:
                    candidates.append({"Ticker": t, "Score": row["mom_6m"]})
            except: pass
            
        df_rank = pd.DataFrame(candidates).sort_values("Score", ascending=False)
        active_tickers = list(df_rank.head(TOP_N)["Ticker"])
        
        # ログ出力（AI SniperとLazy Holdで共通）
        if current.month % 12 == 1 or current.month % 12 == 7:
            print(f"📅 {current.date()} 選抜: {active_tickers}")
        
        if not active_tickers:
            # キャッシュ待機
            period_days = pd.date_range(current, next_rebalance, freq="B")
            for d in period_days:
                equity_curve.append(current_capital)
                dates.append(d)
            current = next_rebalance
            continue

        # 2. 運用 (Micro)
        period_days = pd.date_range(current, next_rebalance, freq="B")
        budget_per_stock = current_capital / len(active_tickers)
        
        # --- Lazy Hold (期初に一括購入) ---
        if mode_name == "Lazy Hold":
            for t in active_tickers:
                try:
                    idx = processed[t].index.get_indexer([current], method='pad')[0]
                    entry_price = processed[t].iloc[idx]["Close"]
                    qty = budget_per_stock / entry_price
                    cost = qty * entry_price * COST_RATE
                    
                    if current_capital >= qty * entry_price + cost:
                        current_capital -= (qty * entry_price + cost)
                        positions[t] = {'qty': qty, 'entry': entry_price, 'stop': 0} # stopは無効
                except: pass

        # 日次ループ
        for d in period_days:
            if d > end: break
            
            # --- AI Sniper (日次売買) ---
            if mode_name == "AI Sniper":
                # Exit Check
                remove_list = []
                for t in list(positions.keys()):
                    if d not in processed[t].index: continue
                    row = processed[t].loc[d]
                    pos = positions[t]
                    
                    # Exit条件: -10%損切り OR SMA20割れ
                    is_exit = False
                    if row['Low'] <= pos['stop']:
                        is_exit = True; exit_p = pos['stop']
                    elif row['Close'] < row['sma20']:
                        is_exit = True; exit_p = row['Close']
                        
                    if is_exit:
                        cash_back = exit_p * pos['qty'] * (1 - COST_RATE)
                        current_capital += cash_back
                        remove_list.append(t)
                
                for t in remove_list: del positions[t]
                
                # Entry Check
                for t in active_tickers:
                    if t in positions: continue
                    if d not in processed[t].index: continue
                    row = processed[t].loc[d]
                    
                    if row['breakout']:
                        cost = budget_per_stock * (1 + COST_RATE)
                        if current_capital >= cost:
                            qty = budget_per_stock / row['Close']
                            current_capital -= (qty * row['Close'] * (1 + COST_RATE))
                            positions[t] = {
                                'qty': qty, 'entry': row['Close'], 
                                'stop': row['Close'] * (1 - STOP_LOSS_PCT)
                            }

            # --- 資産集計 ---
            total_val = current_capital
            
            # Lazy Holdの場合、ポジションがDailyで変わらないので、期末の終値まで保持
            if mode_name == "Lazy Hold":
                for t, pos in positions.items():
                    if d in processed[t].index:
                        price = processed[t].loc[d]["Close"]
                        total_val += price * pos['qty']
                    else:
                        total_val += pos['entry'] * pos['qty']
            
            # AI Sniperの場合、日次の売買でpositionsが変動している
            elif mode_name == "AI Sniper":
                for t, pos in positions.items():
                    if d in processed[t].index:
                        price = processed[t].loc[d]["Close"]
                        total_val += price * pos['qty']
                    else:
                        # エントリーした日以外は、買値で評価しても良いが、より正確には前日終値
                        # 今回はシンプルに、データがない場合は買値で評価 (厳密ではないが影響小)
                        total_val += pos['entry'] * pos['qty']


            equity_curve.append(total_val)
            dates.append(d)
            
        # 期末: 全決済
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

# --- 5. 実行 & 比較 ---
eq_ai, dates_ai = run_strategy("AI Sniper")
eq_hold, dates_hold = run_strategy("Lazy Hold")

# --- 6. 結果評価 ---
def get_metrics(equity):
    s = pd.Series(equity)
    total_ret = (s.iloc[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    # 最大ドローダウン (Max Drawdown)
    dd = (s / s.cummax() - 1).min() * 100
    return int(s.iloc[-1]), total_ret, dd

res_ai = get_metrics(eq_ai)
res_hold = get_metrics(eq_hold)

# 日経平均
nk_norm = (nikkei / nikkei.asof(dates_ai[0])) * INITIAL_CAPITAL
nk_aligned = nk_norm.reindex(dates_ai, method='ffill')
nk_final = nk_aligned.iloc[-1]
if isinstance(nk_final, pd.Series): nk_final = nk_final.item()

res_nk = [int(nk_final), (nk_final - INITIAL_CAPITAL)/INITIAL_CAPITAL*100, (nk_aligned/nk_aligned.cummax()-1).min()*100]

df_res = pd.DataFrame([res_ai, res_hold, res_nk], 
                      index=["🤖 AI Sniper (防御型)", "💪 Lazy Hold (攻撃型)", "🇯🇵 Nikkei 225"],
                      columns=["最終資本", "リターン %", "最大DD %"])

print("\n" + "="*60)
print("【最終決戦結果：1995年からの30年間耐久テスト】")
print(df_res)
print("="*60)

plt.figure(figsize=(12, 6))
plt.plot(dates_ai, eq_ai, label='AI Sniper (防御型)', color='red', linewidth=1.5)
plt.plot(dates_hold, eq_hold, label='Lazy Hold (攻撃型)', color='green', linewidth=2)
plt.plot(dates_ai, nk_aligned, label='Nikkei 225 (市場平均)', color='black', linestyle='--')

# y軸はログスケールにして成長の勢いを比較しやすくする
plt.yscale('log')
plt.title(f"30-Year Backtest: 1995-{END_DATE[:4]} | AI vs Lazy Hold vs Nikkei 225")
plt.ylabel("Capital (JPY - Log Scale)")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()