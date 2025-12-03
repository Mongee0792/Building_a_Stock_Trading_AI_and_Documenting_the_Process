# =========================================================
#  News Score Threshold Test
#  Does "Strong Sentiment" lead to "High Accuracy"?
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 検証ロジック ---
# df_event (ニュースがある日のデータ) を使用
# news_score がプラスなら「上がる(1)」、マイナスなら「下がる(0)」と単純予測して、
# その正解率を閾値ごとに計算する

print("📊 ニューススコアの強度別・勝率分析\n")

thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
results = []

for thr in thresholds:
    # 閾値以上の強いニュースだけ抽出 (絶対値)
    target_data = df_event[df_event['news_score'].abs() >= thr].copy()

    if len(target_data) == 0:
        continue

    # 予測ロジック:
    # スコア > 0 なら「上がる」と予測
    # スコア < 0 なら「下がる」と予測
    # (実際の動き target は 1=上昇, 0=下落)

    # 予測 (1 or 0)
    target_data['pred_dir'] = (target_data['news_score'] > 0).astype(int)

    # 正解数
    correct = (target_data['pred_dir'] == target_data['target']).sum()
    total = len(target_data)
    win_rate = correct / total

    print(f"閾値 {thr:.1f}以上 : 勝率 {win_rate:.2%} (サンプル数: {total})")

    results.append({'Threshold': thr, 'WinRate': win_rate, 'Count': total})

# --- グラフ化 ---
df_res = pd.DataFrame(results)

fig, ax1 = plt.subplots(figsize=(10, 6))

# 勝率 (折れ線)
ax1.plot(df_res['Threshold'], df_res['WinRate'], marker='o', color='red', linewidth=2, label='Win Rate')
ax1.axhline(0.5, color='black', linestyle='--')
ax1.set_xlabel('Sentiment Score Threshold (Absolute)')
ax1.set_ylabel('Win Rate', color='red')
ax1.tick_params(axis='y', labelcolor='red')
ax1.set_ylim(0.4, 0.8) # 40%~80%を表示

# サンプル数 (棒グラフ)
ax2 = ax1.twinx()
ax2.bar(df_res['Threshold'], df_res['Count'], width=0.05, alpha=0.3, color='gray', label='Sample Count')
ax2.set_ylabel('Sample Count', color='gray')

plt.title("Do Stronger News Predict Better?")
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.show()