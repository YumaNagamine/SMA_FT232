import os
from src.utils.logger import Logger

def test_logger_creates_log_file(tmp_path):
    log_dir = tmp_path / "logs"
    logger = Logger(log_dir=str(log_dir))
    logger.info("hello world")
    files = os.listdir(str(log_dir))
    # run_YYYYMMDD-HHMMSS.log が作成されているはず
    assert any(f.startswith("run_") and f.endswith(".log") for f in files)
