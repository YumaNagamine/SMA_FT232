import os
import glob
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt

def generate_plots(
    input_path: str,
    window_size: int = 5,
    output_folder: Optional[str] = None
):
    """
    input_path:      CSVファイルまたはCSVが入ったフォルダ
    window_size:     移動平均のウィンドウ幅
    output_folder:   出力先フォルダ（Noneならinput_pathのあるフォルダ）
    """
    # ── input_path がファイルかフォルダか判定 ──
    if os.path.isfile(input_path):
        csv_files = [input_path]
        base_folder = os.path.dirname(input_path)
    else:
        csv_files = glob.glob(os.path.join(input_path, '*.csv'))
        base_folder = input_path

    # ── 出力先フォルダの決定 ──
    if output_folder is None:
        output_folder = base_folder
    os.makedirs(output_folder, exist_ok=True)

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        base = os.path.splitext(os.path.basename(csv_file))[0]

        # ── 'MCP' 列を x, y に分割 ──
        df[['MCP_x', 'MCP_y']] = (
            df['MCP']
            .str.strip('()')
            .str.split(',', expand=True)
            .astype(float)
        )

        # 1) 移動平均 (angles)
        df_ma = (
            df[['angle0', 'angle1', 'angle2']]
            .rolling(window=window_size, center=True)
            .mean()
        )

        # プロット & 保存 (angles)
        width_px, height_px, dpi = 1080, 720, 100
        fig = plt.figure(figsize=(width_px/dpi, height_px/dpi), dpi=dpi)
        plt.plot(df['time'], df_ma['angle0'], label='angle0')
        plt.plot(df['time'], df_ma['angle1'], label='angle1')
        plt.plot(df['time'], df_ma['angle2'], label='angle2')
        plt.xlabel('time')
        plt.ylabel('angle')
        plt.title(f'{base} – Moving Average (window={window_size})')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'{base}_angles.png'))
        plt.close()

        # 2) 相対散布図（Y軸反転、MCPを中心に）
        a = df['MCP_x']
        b = df['MCP_y']

        # 'marker pos0' → 'fingertip' に変更
        x_f = df['fingertip_x'] - a
        y_f = df['fingertip_y'] - b

        x_d = df['DIP_x'] - a
        y_d = df['DIP_y'] - b

        x_p = df['PIP_x'] - a
        y_p = df['PIP_y'] - b

        fig = plt.figure(figsize=(width_px/dpi, height_px/dpi), dpi=dpi)
        lw = 0.5
        plt.plot(x_f, y_f, label='fingertip', lw=lw)
        plt.plot(x_d, y_d, label='DIP', lw=lw)
        plt.plot(x_p, y_p, label='PIP', lw=lw)
        plt.xlabel('x relative to MCP')
        plt.ylabel('y relative to MCP')
        plt.title(f'{base} – Trajectory relative to MCP')
        plt.axis('equal')
        plt.gca().invert_yaxis()
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'{base}_trajectory.png'))
        plt.close()

if __name__ == '__main__':
    # --- ユーザー設定セクション ---
    core_name = 'slowmo_test'
    folder_path = r'./sc01/' + core_name
    filename = core_name + "_extracted_processed.csv"
    input_path = os.path.join(folder_path, filename)
    window_size = 5      # 後から変更可能
    # -----------------------------
    generate_plots(input_path, window_size)
