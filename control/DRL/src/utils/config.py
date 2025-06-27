import yaml

def load_config(file_path: str) -> dict:
    """
    YAML ファイルを読み込んで辞書として返す
    """
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)