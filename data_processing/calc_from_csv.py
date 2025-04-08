import pandas as pd
import numpy as np
import os

# CSVファイルの読み込み（例: 'data.csv'）
import os
import pandas as pd
import numpy as np

def filter_consecutive_anomalies(df, cols, threshold):
    """
    指定した列群（cols）について、直前の有効な行と比較し、いずれかの成分の差分がthreshold以上の場合はその行を除外する。
    最初の行は必ずキープする。
    """
    # 最初の行は必ずキープ
    keep = [True]
    last_valid = df.iloc[0][cols].astype(float)

    # 2行目以降を順にチェック
    for idx in range(1, len(df)):
        current = df.iloc[idx][cols].astype(float)
        # いずれかの成分で差分が閾値以上なら除外
        if (current - last_valid).abs().gt(threshold).any():
            keep.append(False)
        else:
            keep.append(True)
            last_valid = current  # 現在の値を有効な最新値として更新
    return df[keep].copy()

# 処理対象のフォルダとファイル名を指定
core_name = "FDP_LM_trimed"
folder_path = r'./sc01/'+ core_name 
filename = core_name + "_extracted.csv"

# 入力ファイルのパス
input_path = os.path.join(folder_path, filename)

# CSVファイルの読み込み
df = pd.read_csv(input_path)

# --- marker pos0 カラムの前処理 ---
# もし "marker pos0_x" と "marker pos0_y" が存在しなければ、
# "marker pos0" カラムからそれらを生成する（形式例: "(12.3,45.6)"）
if 'marker pos0_x' not in df.columns or 'marker pos0_y' not in df.columns:
    if 'marker pos0' in df.columns:
        # 括弧を除去してコンマで分割
        marker_coords = df['marker pos0'].str.strip('()').str.split(',', expand=True)
        df['marker pos0_x'] = pd.to_numeric(marker_coords[0], errors='coerce')
        df['marker pos0_y'] = pd.to_numeric(marker_coords[1], errors='coerce')
    else:
        raise KeyError("CSVファイルに 'marker pos0_x' と 'marker pos0_y'、または 'marker pos0' カラムが存在しません。")

# --- DIP, PIP カラムの前処理 ---
# 'DIP' と 'PIP' に "(x,y)" 形式の文字列が入っている前提で，
# それぞれを分割し DIP_x, DIP_y, PIP_x, PIP_y を生成する
for col in ['DIP', 'PIP']:
    x_col = f'{col}_x'
    y_col = f'{col}_y'
    # まだ分割済み列がなければ処理を行う
    if x_col not in df.columns or y_col not in df.columns:
        if col in df.columns:
            # 括弧を除いてコンマで分割
            coords = df[col].str.strip('()').str.split(',', expand=True)
            # 数値変換して新しい列に代入
            df[x_col] = pd.to_numeric(coords[0], errors='coerce')
            df[y_col] = pd.to_numeric(coords[1], errors='coerce')
        else:
            raise KeyError(f"CSVファイルに '{col}' カラムが存在しません。")

# --- エラー測定のデータを除外 ---
# ・angle0が "[]" の行を除外
# ・marker pos0 が (-1, -1) の行を除外
df_clean = df[(df['angle0'] != "[]") & ~((df['marker pos0_x'] == -1) & (df['marker pos0_y'] == -1))].copy()


# --- 角度データの数値変換 ---
# angle0, angle1, angle2が文字列の場合、数値に変換する
for col in ['angle0', 'angle1', 'angle2']:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

# --- 急激な変化の除外（前処理） ---
# 1) 角度: 前フレームとの差分が5度以上の行を除外
angle_diff = df_clean[['angle0','angle1','angle2']].diff().abs()
angle_mask = angle_diff.lt(5).all(axis=1).fillna(True)

# 2) 座標: marker pos0_x または marker pos0_y の差分が200以上の行を除外
coord_diff = df_clean[['marker pos0_x','marker pos0_y']].diff().abs()
coord_mask = coord_diff.lt(1000).all(axis=1).fillna(True)

# 両方の条件を満たす行のみを残す
df_clean = df_clean[angle_mask & coord_mask].copy()

# --- 座標の範囲チェック（0 ≤ x ≤ 1600, 0 ≤ y ≤ 1200）---
# marker pos0_x, marker pos0_y が指定範囲外の行を除去
range_mask = (
    df_clean['marker pos0_x'].between(0, 1600) &
    df_clean['marker pos0_y'].between(0, 1200)
)
df_clean = df_clean[range_mask].copy()
range_mask = (
    df_clean['DIP_x'].between(0, 1600) &
    df_clean['DIP_y'].between(0, 1200)
)
df_clean = df_clean[range_mask].copy()
range_mask = (
    df_clean['PIP_x'].between(0, 1600) &
    df_clean['PIP_y'].between(0, 1200)
)
df_clean = df_clean[range_mask].copy()


# 時刻データ（time列が存在する前提）
time = df_clean['time'].values

# --- 角度の微分計算 ---
# 各角度について1階微分と2階微分を計算
for angle in ['angle0', 'angle1', 'angle2']:
    angle_data = df_clean[angle].values
    dangle_dt = np.gradient(angle_data, time)       # 1階微分: d(angle)/dt
    d2angle_dt2 = np.gradient(dangle_dt, time)        # 2階微分: d²(angle)/dt²
    df_clean[f'd{angle}_dt'] = dangle_dt
    df_clean[f'd2{angle}_dt2'] = d2angle_dt2

# --- marker pos0 の座標データから速度の計算 ---
x = df_clean['marker pos0_x'].values
y = df_clean['marker pos0_y'].values
dx_dt = np.gradient(x, time)
dy_dt = np.gradient(y, time)
speed = np.sqrt(dx_dt**2 + dy_dt**2)
df_clean['speed'] = speed

# --- 出力ファイル名の生成 ---
base, ext = os.path.splitext(filename)
output_filename = f"{base}_processed{ext}"
output_path = os.path.join(folder_path, output_filename)

# 結果を新しいCSVファイルに保存
df_clean.to_csv(output_path, index=False)

print(f"解析結果を {output_path} に保存しました。")