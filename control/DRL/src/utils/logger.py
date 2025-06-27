import os
from datetime import datetime

class Logger:
    """
    シンプルなファイル＆コンソールログ出力クラス
    """
    def __init__(self, log_dir="logs"):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_file = os.path.join(log_dir, f"run_{timestamp}.log")
        os.makedirs(log_dir, exist_ok=True)

    def info(self, message: str):
        line = f"[INFO] {datetime.now().isoformat()} - {message}"
        print(line)
        with open(self.log_file, 'a') as f:
            f.write(line + "\n")

    def error(self, message: str):
        line = f"[ERROR] {datetime.now().isoformat()} - {message}"
        print(line)
        with open(self.log_file, 'a') as f:
            f.write(line + "\n")
