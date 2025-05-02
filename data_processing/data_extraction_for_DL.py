import pandas as pd

def create_annotations(input_csv_path: str, output_csv_path: str, video_name: str):
    """
    xxx_extracted_processed.csv -> annotations.csv
    save videoname, frame number, each joint pos

    """
    # CSVの読み込み
    df = pd.read_csv(input_csv_path)

    # marker pos0 を分割して fingertip_x, fingertip_y を作成
    pos0 = df['marker pos0'].str.strip('()').str.split(',', expand=True)
    df['fingertip_x'] = pos0[0].astype(float)
    df['fingertip_y'] = pos0[1].astype(float)

    # modified marker pos6 を分割して MCP_x, MCP_y を作成
    pos6 = df['modified marker pos6'].str.strip('()').str.split(',', expand=True)
    df['MCP_x'] = pos6[0].astype(float)
    df['MCP_y'] = pos6[1].astype(float)

    # videoname 列を追加
    df['videoname'] = video_name

    # 必要な列を選択して順序を指定
    output_df = df[[
        'videoname',
        'frame',
        'fingertip_x', 'fingertip_y',
        'DIP_x', 'DIP_y',
        'PIP_x', 'PIP_y',
        'MCP_x', 'MCP_y'
    ]]

    # CSVとして保存
    output_df.to_csv(output_csv_path, index=False)
    print(f"Saved annotations to {output_csv_path}")

if __name__ == "__main__":

    video_name = "FDP"
    input_csv = "./sc01/" + video_name + "/"+ video_name + "_extracted_processed.csv"
    output_csv = "./sc01/" + video_name + "/annotations.csv"

    create_annotations(input_csv, output_csv, video_name)
