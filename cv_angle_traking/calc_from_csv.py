import pandas as pd
import numpy as np
import os

# CSVファイルの読み込み（例: 'data.csv'）
import os
import pandas as pd
import numpy as np

# 処理対象のフォルダとファイル名を指定
core_name = "output_to_analyze"
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

# --- エラー測定のデータを除外 ---
# ・angle0が "[]" の行を除外
# ・marker pos0 が (-1, -1) の行を除外
df_clean = df[(df['angle0'] != "[]") & ~((df['marker pos0_x'] == -1) & (df['marker pos0_y'] == -1))].copy()

# --- 角度データの数値変換 ---
# angle0, angle1, angle2が文字列の場合、数値に変換する
for col in ['angle0', 'angle1', 'angle2']:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

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