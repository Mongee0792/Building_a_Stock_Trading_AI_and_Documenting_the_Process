# =========================================================
#  FINAL BATTLE: NVDA 10-Year News Analysis
#  Target: Verify News Importance with Recovered Data
# =========================================================

import pandas as pd
import numpy as np
import talib
import lightgbm as lgb
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
from tqdm import tqdm
import torch
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import os
import sys

## --- 1. ニュースデータの読み込み (修正版) ---
CSV_FILE = "nvda_news_fixed.csv"

print(f"📂 '{CSV_FILE}' を読み込み中...")
if not os.path.exists(CSV_FILE):
    print("❌ エラー: 修正済みCSVが見つかりません。前のステップを実行してください。")
    sys.exit()

df_news = pd.read_csv(CSV_FILE)

# ★修正点: format='mixed' であらゆる形式に対応させる
df_news['date'] = pd.to_datetime(df_news['date'], format='mixed', utc=True).dt.tz_localize(None)

print(f"✅ ニュース数: {len(df_news)} 件")
print(f"📅 期間: {df_news['date'].min().date()} 〜 {df_news['date'].max().date()}")

# --- 以降は変更なし ---

# --- 2. FinBERTで全件スコアリング (GPU推奨) ---
print("\n🧠 AI (FinBERT) が10年分のニュースを採点中...")

device = 0 if torch.cuda.is_available() else -1
print(f"  Device: {'GPU' if device == 0 else 'CPU'}")

classifier = pipeline("text-classification", model="ProsusAI/finbert", device=device, top_k=None)

sentiment_scores = []
headlines = df_news['headline'].tolist()

# バッチ処理 (64件ずつ)
batch_size = 64
for i in tqdm(range(0, len(headlines), batch_size), desc="Scoring"):
    batch = headlines[i : i+batch_size]
    try:
        batch_clean = [str(text)[:512] for text in batch] # 512文字制限
        results = classifier(batch_clean)

        for res in results:
            score_dict = {x['label']: x['score'] for x in res}
            score = score_dict.get('positive', 0) - score_dict.get('negative', 0)
            sentiment_scores.append(score)
    except:
        sentiment_scores.extend([0] * len(batch))

df_news['news_score'] = sentiment_scores

# 日次集計 (同日のニュースは平均)
daily_sentiment = df_news.groupby('date')['news_score'].mean()

# --- 3. 株価データ取得 (安全版) ---
TARGET_TICKER = "NVDA"

# 日付範囲の確認
if not df_news.empty:
    start_dt = df_news['date'].min()
    end_dt = df_news['date'].max() + pd.Timedelta(days=5)
    print(f"📅 ニュース期間: {start_dt.date()} 〜 {end_dt.date()}")
else:
    # ニュースがない場合のバックアップ期間
    start_dt = "2011-01-01"
    end_dt = "2021-01-01"
    print("⚠️ ニュースがないため、デフォルト期間を使用します。")

print(f"📊 {TARGET_TICKER} の株価を取得中...")

try:
    # 株価取得
    df_price_raw = yf.download(TARGET_TICKER, start=start_dt, end=end_dt, interval="1d", progress=False)

    # データ確認
    if df_price_raw.empty:
        raise ValueError("株価データが空です。期間または銘柄コードを確認してください。")

    # カラム整形 (MultiIndex対策)
    if isinstance(df_price_raw.columns, pd.MultiIndex):
        # Close列だけをSeriesとして抽出
        df_price = df_price_raw['Close'].iloc[:, 0] if df_price_raw['Close'].shape[1] > 0 else df_price_raw['Close']
        # Volumeも同様に
        df_vol = df_price_raw['Volume'].iloc[:, 0] if df_price_raw['Volume'].shape[1] > 0 else df_price_raw['Volume']

        # High/Lowも
        df_high = df_price_raw['High'].iloc[:, 0]
        df_low = df_price_raw['Low'].iloc[:, 0]

    else:
        df_price = df_price_raw['Close']
        df_vol = df_price_raw['Volume']
        df_high = df_price_raw['High']
        df_low = df_price_raw['Low']

    # タイムゾーン削除
    df_price.index = df_price.index.tz_localize(None)
    df_vol.index = df_vol.index.tz_localize(None)
    df_high.index = df_high.index.tz_localize(None)
    df_low.index = df_low.index.tz_localize(None)

    # 結合 (DataFrame作成)
    df = pd.DataFrame({
        'Close': df_price,
        'High': df_high,
        'Low': df_low,
        'Volume': df_vol
    })

    # ニューススコアのマージ
    # daily_sentiment がある場合のみ
    if 'daily_sentiment' in locals():
        df['news_score'] = df.index.map(daily_sentiment)
        df['news_score'] = df['news_score'].fillna(0) # ニュースなし日は0

        # ニュースあり率の確認
        non_zero = (df['news_score'] != 0).sum()
        print(f"✅ ニュース反映日数: {non_zero} 日 / 全 {len(df)} 日")
    else:
        print("⚠️ daily_sentiment が見つかりません。ニューススコアは全て0になります。")
        df['news_score'] = 0

    # --- 4. 特徴量エンジニアリング & 検証 ---
    print("⚙️ AI学習準備...")

    # テクニカル指標
    df['rsi'] = talib.RSI(df['Close'], 14)
    df['adx'] = talib.ADX(df['High'], df['Low'], df['Close'], 14)
    df['vol_change'] = df['Volume'].pct_change()
    df['return_1d'] = df['Close'].pct_change()

    # Target: 翌日上がるか？
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    df.dropna(inplace=True)
    print(f"学習データセット: {len(df)} 行")

    # LightGBM 検証
    features = ['rsi', 'adx', 'return_1d', 'vol_change', 'news_score']
    tscv = TimeSeriesSplit(n_splits=5)
    importances = pd.DataFrame(index=features)
    acc_scores = []

    print("\n=== バックテスト開始 ===")
    for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
        X_train, y_train = df.iloc[train_idx][features], df.iloc[train_idx]['target']
        X_test, y_test = df.iloc[test_idx][features], df.iloc[test_idx]['target']

        model = lgb.LGBMClassifier(random_state=42, verbose=-1, n_estimators=100)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        acc_scores.append(acc)
        importances[f'Fold_{fold}'] = model.feature_importances_

        print(f"Fold {fold+1}: 正解率 {acc:.2%}")

    print(f"\n平均正解率: {np.mean(acc_scores):.2%}")

    # 重要度可視化
    importances['Average'] = importances.mean(axis=1)
    importances = importances.sort_values('Average', ascending=False)

    print("\n【特徴量重要度ランキング】")
    print(importances['Average'])

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances['Average'], y=importances.index, palette='viridis')
    plt.title(f"Feature Importance: {TARGET_TICKER} (10-Year News Impact)")
    plt.xlabel("Importance")
    plt.grid(axis='x')
    plt.show()

except Exception as e:
    print(f"❌ 処理エラー: {e}")
    import traceback
    traceback.print_exc()







# =========================================================
#  Event-Driven Test: Does News Matter on "News Days"?
#  Target: Only rows where news_score != 0
# =========================================================

print("\n=== 📰 イベント・ドリブン検証 (ニュースがある日限定) ===")

# ニュースがある日だけ抽出
df_event = df[df['news_score'] != 0].copy()
print(f"検証データ数: {len(df_event)} 行")

# LightGBMで再検証
features = ['rsi', 'adx', 'return_1d', 'vol_change', 'news_score']
tscv = TimeSeriesSplit(n_splits=5)
importances_event = pd.DataFrame(index=features)
acc_scores_event = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(df_event)):
    X_train, y_train = df_event.iloc[train_idx][features], df_event.iloc[train_idx]['target']
    X_test, y_test = df_event.iloc[test_idx][features], df_event.iloc[test_idx]['target']

    model = lgb.LGBMClassifier(random_state=42, verbose=-1, n_estimators=100)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    acc_scores_event.append(acc)
    importances_event[f'Fold_{fold}'] = model.feature_importances_
    print(f"Fold {fold+1}: 正解率 {acc:.2%}")

print(f"\n平均正解率: {np.mean(acc_scores_event):.2%}")

# 重要度可視化
importances_event['Average'] = importances_event.mean(axis=1)
importances_event = importances_event.sort_values('Average', ascending=False)

print("\n【イベント日限定・重要度ランキング】")
print(importances_event['Average'])

plt.figure(figsize=(10, 6))
sns.barplot(x=importances_event['Average'], y=importances_event.index, palette='magma')
plt.title(f"Feature Importance (News Days Only)")
plt.xlabel("Importance")
plt.grid(axis='x')
plt.show()