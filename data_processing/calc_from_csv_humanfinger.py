import os
import pandas as pd
import numpy as np

def filter_consecutive_anomalies(df, cols, threshold):
    """
    指定した列群（cols）について、直前の有効な行と比較し、
    いずれかの成分の差分が threshold 以上の場合はその行を除外する。
    最初の行は必ずキープする。
    """
    keep = [True]
    last_valid = df.iloc[0][cols].astype(float)
    for idx in range(1, len(df)):
        current = df.iloc[idx][cols].astype(float)
        if (current - last_valid).abs().gt(threshold).any():
            keep.append(False)
        else:
            keep.append(True)
            last_valid = current
    return df[keep].copy()

# --- 設定 ---
core_name   = "slowmo_test"
folder_path = os.path.join('.', 'sc01', core_name)
filename    = core_name + "_extracted.csv"
input_path  = os.path.join(folder_path, filename)

# --- CSV読み込み ---
df = pd.read_csv(input_path)

# --- fingertip の前処理 ---
if 'fingertip_x' not in df.columns or 'fingertip_y' not in df.columns:
    if 'fingertip' in df.columns:
        coords = df['fingertip'].str.strip('()').str.split(',', expand=True)
        df['fingertip_x'] = pd.to_numeric(coords[0], errors='coerce')
        df['fingertip_y'] = pd.to_numeric(coords[1], errors='coerce')
    else:
        raise KeyError("CSVに 'fingertip_x','fingertip_y' または 'fingertip' カラムがありません。")

# --- DIP, PIP の前処理 ---
for col in ['DIP', 'PIP']:
    x_col = f'{col}_x'
    y_col = f'{col}_y'
    if x_col not in df.columns or y_col not in df.columns:
        if col in df.columns:
            coords = df[col].str.strip('()').str.split(',', expand=True)
            df[x_col] = pd.to_numeric(coords[0], errors='coerce')
            df[y_col] = pd.to_numeric(coords[1], errors='coerce')
        else:
            raise KeyError(f"CSVに '{col}' カラムがありません。")

# --- 初期エラー値の除去 ---
df_clean = df[
    (df['angle0'] != "[]") &
    ~((df['fingertip_x'] == -1) & (df['fingertip_y'] == -1))
].copy()

# --- 角度データの数値変換 ---
for col in ['angle0', 'angle1', 'angle2']:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

# --- 角度の外れ値除去（連続的な測定エラー） ---
angle_cols      = ['angle0', 'angle1', 'angle2']
angle_threshold = 30   # 30°以上のジャンプを外れ値とみなす（調整可）
df_clean = filter_consecutive_anomalies(df_clean, angle_cols, angle_threshold)

# --- 座標の外れ値除去（連続的な測定エラー） ---
position_cols = ['fingertip_x', 'fingertip_y']
threshold     = 100  # ピクセル単位の閾値（調整可）
df_clean = filter_consecutive_anomalies(df_clean, position_cols, threshold)

# --- 座標の範囲チェック (0 ≤ x ≤ 1600, 0 ≤ y ≤ 1200) ---
for col_x, col_y in [
    ('fingertip_x','fingertip_y'),
    ('DIP_x','DIP_y'),
    ('PIP_x','PIP_y')
]:
    mask = df_clean[col_x].between(0, 1600) & df_clean[col_y].between(0, 1200)
    df_clean = df_clean[mask].copy()

# --- 時刻データ取得 ---
time = df_clean['time'].values

# --- 角度の微分計算 ---
for angle in ['angle0', 'angle1', 'angle2']:
    angle_data    = df_clean[angle].values
    dangle_dt     = np.gradient(angle_data, time)
    d2angle_dt2   = np.gradient(dangle_dt, time)
    df_clean[f'd{angle}_dt']   = dangle_dt
    df_clean[f'd2{angle}_dt2'] = d2angle_dt2

# --- fingertip の速度計算 ---
x     = df_clean['fingertip_x'].values
y     = df_clean['fingertip_y'].values
dx_dt = np.gradient(x, time)
dy_dt = np.gradient(y, time)
df_clean['speed'] = np.sqrt(dx_dt**2 + dy_dt**2)

# --- 出力ファイル保存 ---
base, ext       = os.path.splitext(filename)
output_filename = f"{base}_processed{ext}"
output_path     = os.path.join(folder_path, output_filename)
df_clean.to_csv(output_path, index=False)

print(f"解析結果を {output_path} に保存しました。")
