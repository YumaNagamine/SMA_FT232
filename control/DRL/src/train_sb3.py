import yaml
from stable_baselines3 import SAC
from src.env.finger_gym import FingerGymEnv

def main():
    # 設定ファイル読み込み
    with open('configs/env.yaml') as f:
        env_cfg = yaml.safe_load(f)
    with open('configs/reward_improved.yaml') as f:
        reward_cfg = yaml.safe_load(f)
    cfg = {'env': env_cfg, 'reward': reward_cfg, 'hw': {'port':'/dev/ttyUSB0'}}

    env = FingerGymEnv(cfg, device='gpu')
    model = SAC('MlpPolicy', env,
                learning_rate=3e-4,
                buffer_size=1_000_000,
                batch_size=128,
                gamma=0.99,
                tau=0.005,
                train_freq=50,
                gradient_steps=1,
                verbose=1,
                tensorboard_log="./sb3_tb/")
    
    # 学習開始
    model.learn(total_timesteps=1_000_000)
    # 保存
    model.save("models/sb3_sac_finger")
    
if __name__ == '__main__':
    main()
