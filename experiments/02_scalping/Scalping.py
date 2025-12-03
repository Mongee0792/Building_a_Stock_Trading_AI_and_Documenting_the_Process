# =========================================================
#  BTC 5m 成行、指値ハイブリッド注文 (詳細分析モード,手数料の値も掲載)
#  Breakdown: Win Rate, Fees, PnL by Order Type
# =========================================================

import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt

# (データ読み込みと特徴量エンジニアリングは完了している前提で進めます)
# もし `full_df` が消えていたら、前の「データ読み込みコード」を再実行してください。

# --- 3. 検証設定 ---
n_folds = 5
THRESHOLD = 0.62
BET_AMOUNT = 200_000

# 手数料設定 (Binance Futures VIP0想定)
MAKER_FEE_RATE = 0.0002  # 0.02%
TAKER_FEE_RATE = 0.0005  # 0.05%
LIMIT_OFFSET = 0.0005    # -0.05%

# モデル学習・推論関数 (変更なし)
def train_ensemble(X, y):
    lgb_m = lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1, n_estimators=300)
    lgb_m.fit(X, y)
    xgb_m = xgb.XGBClassifier(random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='logloss', n_estimators=300)
    xgb_m.fit(X, y)
    cat_m = CatBoostClassifier(random_state=42, verbose=0, allow_writing_files=False, iterations=300)
    cat_m.fit(X, y)
    return [lgb_m, xgb_m, cat_m]

def predict_ensemble(models, X):
    p1 = models[0].predict_proba(X)[:, 1]
    p2 = models[1].predict_proba(X)[:, 1]
    p3 = models[2].predict_proba(X)[:, 1]
    return (p1 + p2 + p3) / 3

# 特徴量リスト
features = [
    'rsi', 'bb_width', 'bb_pos', 'roc_short', 'roc_mid',
    'trend_div', 'adx', 'atr_ratio', 'vol_ratio',
    'hour', 'dayofweek', 'meta_long'
]

# --- 4. ハイブリッド検証実行 (詳細版) ---
total_rows = len(full_df)
fold_size = int(total_rows / (n_folds + 1))
equity_curve = [1_000_000]
current_capital = 1_000_000

print(f"\n=== BTC 5m ハイブリッド注文 (詳細分析モード) ===")

for i in range(n_folds):
    train_end = fold_size * (i + 1)
    test_end = fold_size * (i + 2)

    train_subset = full_df.iloc[:train_end]
    test_subset = full_df.iloc[train_end:test_end]

    print(f"\nPeriod {i+1}:")

    # 戦略: RSI < 30 逆張り
    train_L = train_subset[train_subset['rsi'] < 30]
    test_L = test_subset[test_subset['rsi'] < 30]

    # 集計用変数
    stats = {
        'limit':  {'wins': 0, 'count': 0, 'pnl': 0, 'fees': 0},
        'market': {'wins': 0, 'count': 0, 'pnl': 0, 'fees': 0},
        'missed': 0
    }

    if len(train_L) > 100 and len(test_L) > 0:
        models_L = train_ensemble(train_L[features], train_L['target_long'])
        probs_L = predict_ensemble(models_L, test_L[features])

        test_L_reset = test_L.reset_index(drop=True)

        for j in range(len(probs_L)):
            if probs_L[j] >= THRESHOLD:
                current_price = test_L_reset['close'].iloc[j]
                limit_price = current_price * (1 - LIMIT_OFFSET)

                next_low = test_L_reset['next_low'].iloc[j]
                next_close = test_L_reset['next_close'].iloc[j]
                exit_price = test_L_reset['exit_price'].iloc[j]

                # --- 1. 指値判定 (Maker) ---
                if next_low <= limit_price:
                    stats['limit']['count'] += 1

                    # 利益計算 (ROI)
                    raw_roi = (exit_price - limit_price) / limit_price

                    # 勝敗カウント (手数料引く前の純粋な勝ち負け)
                    if raw_roi > 0: stats['limit']['wins'] += 1

                    # 手数料 (往復分と仮定: Entry Maker + Exit Taker or Maker?)
                    # ここでは保守的に「Exitは成行(Taker)」として計算するのが安全
                    # Entry(Maker) + Exit(Taker) = 0.02% + 0.05% = 0.07%
                    # ※指値エントリーのメリットは「安く買えること」に集中させる
                    fee_amount = BET_AMOUNT * (MAKER_FEE_RATE + TAKER_FEE_RATE)

                    profit = (BET_AMOUNT * raw_roi) - fee_amount
                    stats['limit']['pnl'] += profit
                    stats['limit']['fees'] += fee_amount

                # --- 2. 成行追いかけ判定 (Taker) ---
                elif next_close > current_price:
                    stats['market']['count'] += 1

                    # 取得単価は次の足の終値 (高値掴み)
                    entry_price = next_close
                    raw_roi = (exit_price - entry_price) / entry_price

                    if raw_roi > 0: stats['market']['wins'] += 1

                    # 手数料 (往復Taker) = 0.05% + 0.05% = 0.1%
                    fee_amount = BET_AMOUNT * (TAKER_FEE_RATE + TAKER_FEE_RATE)

                    profit = (BET_AMOUNT * raw_roi) - fee_amount
                    stats['market']['pnl'] += profit
                    stats['market']['fees'] += fee_amount

                # --- 3. スルー ---
                else:
                    stats['missed'] += 1

        # --- 詳細結果出力 ---
        def print_stats(label, s):
            win_rate = s['wins'] / s['count'] if s['count'] > 0 else 0
            print(f"  [{label}]")
            print(f"    > 勝率: {win_rate:.1%} ({s['wins']}/{s['count']})")
            print(f"    > 損益: {int(s['pnl']):+,} 円")
            print(f"    > 手数料: -{int(s['fees']):,} 円")

        print_stats("指値 (Maker)", stats['limit'])
        print_stats("成行 (Chase)", stats['market'])
        print(f"  [Missed] {stats['missed']} 回")

        period_total = stats['limit']['pnl'] + stats['market']['pnl']
        current_capital += period_total

    else:
        print("  > チャンスなし")

    equity_curve.append(current_capital)
    print(f"  -----------------------------")
    print(f"  期間合計損益: {int(current_capital - equity_curve[-2]):+,} 円")

print("\n" + "="*40)
print(f"Final Capital: {int(current_capital):,} 円")

plt.figure(figsize=(12, 6))
plt.plot(equity_curve, marker='o', label='Hybrid Detailed', color='purple', linewidth=2)
plt.axhline(y=1_000_000, color='red', linestyle='--')
plt.title("BTC Hybrid Strategy: Detailed Breakdown")
plt.grid(True)
plt.show()