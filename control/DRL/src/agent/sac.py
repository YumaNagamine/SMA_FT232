import numpy as np

class SACAgent:
    """
    SACアルゴリズムのエージェント骨組み（最小依存版）
    """
    def __init__(self, config):
        """
        config: dict で学習ハイパーパラメータ等を受け取る想定
        """
        self.config = config

    def select_action(self, state, evaluate=False):
        """
        状態 state に対して行動を返すダミー実装。
        実際はネットワーク推論等を行う。
        """
        # 6次元（SMAワイヤ6本分）の Duty 比を返す
        return np.zeros(6, dtype=float)

    def store_transition(self, state, action, reward, next_state, done):
        """
        リプレイバッファへの保存を行う想定メソッド
        """
        pass

    def train(self):
        """
        バッファからサンプリングし、ネットワーク更新を行う想定メソッド
        """
        pass
