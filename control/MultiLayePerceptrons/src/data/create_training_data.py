# integrate csv data to a single csv file, to make training data

import glob
import os
import pandas as pd

def find_final_row(df, angle_cols, threshold=1):
    """
    各行の角度差分（前行との差の絶対値）がすべて threshold 以下になる
    最初の行番号を返す。該当行がなければ最終行を返す。
    """
    # angle_cols 列の差分を計算
    diffs = df[angle_cols].diff().abs()
    # 閾値以下かどうかを判定（全カラムで True の行を探す）
    mask = (diffs < threshold).all(axis=1)
    # 1 行目（index=0）は初期状態なので除外
    candidates = mask[mask & (mask.index != 0)]
    if not candidates.empty:
        return candidates.index[0]
    else:
        return df.index[-1]

# カレントディレクトリのすべての .csv ファイルを取得
target_dir = "./"
csv_pattern = os.path.join(target_dir, "*.csv")
csv_files = glob.glob(csv_pattern)

records = []
for path in csv_files:
    df = pd.read_csv(path)
    # angle0, angle1, …, angle_top を自動検出
    angle_cols = [c for c in df.columns if c.startswith("angle")]
    
    # 初期状態
    init = df.iloc[0]
    # 最終状態（角度差分がほとんどなくなる行）
    final_idx = find_final_row(df, angle_cols, threshold=1e-3)
    final = df.loc[final_idx]
    
    # ファイル名を first column にして、２行分レコードを作成
    for state_label, row in (("initial", init), ("final", final)):
        rec = {"file_name": os.path.basename(path), "state": state_label}
        rec.update(row.to_dict())
        records.append(rec)

# DataFrame にまとめて CSV 出力
out_df = pd.DataFrame(records)
out_df.to_csv("summary.csv", index=False)

print("Done: summary.csv を出力しました。")
