# =========================================================
#  BTC 5m "Expected Value" Scalper (Regression)
#  Logic: Predict Return -> If > Fee, Deploy Grid
#  Target: High Frequency Trading (No RSI Filter)
# =========================================================

import sys
import pandas as pd
import numpy as np
import talib
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
import glob
import os
from google.colab import drive

# --- 1. データ準備 ---
drive.mount('/content/drive')
parquet_path = "/content/drive/MyDrive/CryptoData/BTC_USDT_5m.parquet"

if not os.path.exists(parquet_path):
    print("❌ データなし")
else:
    print("データ読み込み中...")
    df = pd.read_parquet(parquet_path)

# --- 2. 特徴量生成 (回帰用) ---
def process_regression_data(df):
    print("特徴量生成中 (回帰Ver)...")
    df = df.copy()

    # テクニカル
    df['rsi'] = talib.RSI(df['close'], 14)
    u, m, l = talib.BBANDS(df['close'], 20, 2, 2)
    df['bb_width'] = (u - l) / m
    df['bb_pos'] = (df['close'] - l) / (u - l)

    df['roc_short'] = df['close'].pct_change(3)
    df['roc_mid'] = df['close'].pct_change(12)

    sma200 = talib.SMA(df['close'], 200)
    df['trend_div'] = (df['close'] - sma200) / sma200

    df['adx'] = talib.ADX(df['high'], df['low'], df['close'], 14)
    df['atr'] = talib.ATR(df['high'], df['low'], df['close'], 14)
    df['atr_ratio'] = df['atr'] / df['close']

    vol_sma = talib.SMA(df['volume'].astype(float), 20)
    df['vol_ratio'] = df['volume'] / vol_sma

    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek

    # 指値判定用
    df['next_low'] = df['low'].shift(-1)
    df['exit_price_1h'] = df['close'].shift(-3) # 15分後(3本後)

    # ★Target: 15分後の「リターン(%)」そのものを予測
    df['target_return'] = df['close'].shift(-3) / df['close'] - 1

    # メタラベリング (直近24時間のボラティリティ傾向などを入れる)
    # ここではシンプルに「直近24時間のリターン合計」を入れる
    df['meta_mom'] = df['close'].pct_change(288).shift(1)

    df.dropna(inplace=True)
    return df.reset_index(drop=True)

full_df = process_regression_data(df)
print(f"学習データ数: {len(full_df)} 行")

# --- 3. 検証設定 ---
n_folds = 5
BET_AMOUNT = 200_000

# ★攻撃的設定★
# AI予測リターンが「0.08%」を超えたらGO
# (手数料0.05% + スプレッド分を考慮してプラス期待値なら行く)
PRED_THRESHOLD = 0.0008

# グリッド設定 (浅めに広く取る)
# 予測が良いなら、浅い押し目でも拾いたい
GRID_OFFSETS = [0.0002, 0.0010, 0.0020] # -0.02%, -0.1%, -0.2%
GRID_WEIGHTS = [0.3, 0.3, 0.4]

FEE_MAKER = 0.0002
FEE_TAKER = 0.0005

features = [
    'rsi', 'bb_width', 'bb_pos', 'roc_short', 'roc_mid',
    'trend_div', 'adx', 'atr_ratio', 'vol_ratio',
    'hour', 'dayofweek', 'meta_mom'
]

# --- 4. WFA実行 (回帰モデル) ---
tscv = TimeSeriesSplit(n_splits=n_folds)
equity_curve = [1_000_000]
current_capital = 1_000_000

print(f"\n=== 期待値スキャルピング (閾値 {PRED_THRESHOLD*100:.2f}%) ===")

fold = 1
for train_idx, test_idx in tscv.split(full_df):
    train_df = full_df.iloc[train_idx]
    test_df = full_df.iloc[test_idx].reset_index(drop=True)

    print(f"\nPeriod {fold}: {test_df['timestamp'].iloc[0]} 〜")

    # ★回帰モデル (LGBMRegressor)
    model = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1, n_estimators=200)
    model.fit(train_df[features], train_df['target_return'])

    # 予測 (確信度ではなく「リターン」が出てくる)
    pred_rets = model.predict(test_df[features])

    period_pnl = 0
    total_orders = 0
    total_fills = 0

    # シミュレーション
    for i in range(len(test_df) - 3):
        # 予測リターンが閾値を超えているか？
        # RSIなどのフィルタは一切なし！AIの予測のみ！
        if pred_rets[i] > PRED_THRESHOLD:

            current_price = test_df['close'].iloc[i]
            next_low = test_df['next_low'].iloc[i]
            exit_price = test_df['exit_price_1h'].iloc[i] # 15分後

            # グリッド展開
            for offset, weight in zip(GRID_OFFSETS, GRID_WEIGHTS):
                total_orders += 1
                limit_price = current_price * (1 - offset)

                # 約定判定
                if next_low <= limit_price:
                    qty = (BET_AMOUNT * weight) / limit_price
                    gross_pnl = (exit_price - limit_price) * qty
                    fee = (limit_price * qty * FEE_MAKER) + (exit_price * qty * FEE_TAKER)

                    period_pnl += (gross_pnl - fee)
                    total_fills += 1

    current_capital += period_pnl

    # 統計
    fill_rate = total_fills / total_orders * 100 if total_orders > 0 else 0
    daily_trades = total_fills / (len(test_df) / 288)

    print(f"  > 損益: {int(period_pnl):+,} 円")
    print(f"  > 約定: {total_fills}回 (1日平均 {daily_trades:.1f}回) / 約定率 {fill_rate:.1f}%")

    equity_curve.append(current_capital)
    print(f"  残高: {int(current_capital):,} 円")
    fold += 1

# --- グラフ ---
print("\n" + "="*40)
print(f"Final Capital: {int(current_capital):,} 円")

plt.figure(figsize=(12, 6))
plt.plot(equity_curve, marker='o', label='Expected Value Scalper', color='magenta', linewidth=2)
plt.axhline(y=1_000_000, color='red', linestyle='--')
plt.title("BTC High-Frequency Regression Scalper")
plt.grid(True)
plt.show()





import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# --- 1. 予測精度チェック (散布図) ---
print("予測精度を分析中...")

# 直近のテストデータで予測
preds = model.predict(test_df[features])
actuals = test_df['target_return']

# データフレーム化
df_scatter = pd.DataFrame({
    'Predicted_Return': preds,
    'Actual_Return': actuals
})

# ランダムにサンプリング (点が多すぎると重いので5000点だけ)
df_sample = df_scatter.sample(n=min(5000, len(df_scatter)), random_state=42)

plt.figure(figsize=(10, 10))
sns.scatterplot(x='Predicted_Return', y='Actual_Return', data=df_sample, alpha=0.3)

# ゼロラインと回帰線
plt.axhline(0, color='black', linestyle='--')
plt.axvline(0, color='black', linestyle='--')
plt.plot([-0.01, 0.01], [-0.01, 0.01], color='red', linestyle=':', label='Perfect Prediction')

# 閾値ライン (ここより右がエントリー対象)
plt.axvline(PRED_THRESHOLD, color='green', linestyle='-', label='Entry Threshold')

plt.title(f"AI Prediction vs Reality (Correlation: {df_scatter.corr().iloc[0,1]:.3f})")
plt.xlabel("AI Predicted Return")
plt.ylabel("Actual Return (15min later)")
plt.legend()
plt.grid(True)
plt.show()

# --- 2. トレードポイント可視化 (直近3日分) ---
print("トレード箇所をチャートに描画中...")

# 直近3日分 (288 * 3 = 864本) を切り出し
plot_len = 864
if len(test_df) < plot_len: plot_len = len(test_df)
plot_df = test_df.iloc[-plot_len:].reset_index(drop=True)
plot_preds = preds[-plot_len:]

# トレード抽出
entries = []
exits = []
pnls = []

for i in range(len(plot_df) - 3):
    # エントリー条件: 予測リターン > 閾値
    if plot_preds[i] > PRED_THRESHOLD:
        current_price = plot_df['close'].iloc[i]

        # グリッド1段目 (簡易)
        limit_price = current_price * (1 - GRID_OFFSETS[0])
        next_low = plot_df['next_low'].iloc[i]

        if next_low <= limit_price:
            # 約定
            exit_price = plot_df['exit_price_1h'].iloc[i] # 実際は3本後(15分)
            # 描画用にインデックス保存
            entries.append((i, limit_price))
            exits.append((i+3, exit_price))

            # 勝敗判定 (手数料0.1%込)
            pnl = (exit_price - limit_price) / limit_price - (FEE_MAKER + FEE_TAKER)
            pnls.append(pnl)

# チャート描画
plt.figure(figsize=(16, 8))
plt.plot(plot_df.index, plot_df['close'], color='gray', alpha=0.5, label='Price')

# エントリー (▲)
if entries:
    e_idx, e_price = zip(*entries)
    # 勝ちトレードは緑、負けは赤で色分けしたいが、簡易的に青で統一
    # (個別に色分けすると重くなるため)

    # 勝敗で色分け
    colors = ['lime' if p > 0 else 'red' for p in pnls]
    plt.scatter(e_idx, e_price, c=colors, marker='^', s=80, zorder=5, label='Entry (Green=Win, Red=Loss)')

# エグジット (x)
if exits:
    x_idx, x_price = zip(*exits)
    plt.scatter(x_idx, x_price, marker='x', color='black', s=30, zorder=5, label='Exit (15min)')

# 予測値もサブプロットで表示したいが、まずはメインチャートで見やすく
plt.title("Trade Visualization (Last 3 Days)")
plt.legend()
plt.grid(True)
plt.show()

print(f"表示期間内のトレード数: {len(entries)}回")
if len(pnls) > 0:
    win_rate = sum([1 for p in pnls if p > 0]) / len(pnls)
    print(f"期間内勝率: {win_rate:.1%}")