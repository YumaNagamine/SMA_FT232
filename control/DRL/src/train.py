import argparse
from src.utils.config import load_config
from src.reward.reward_fn_user import RewardUser
from src.reward.reward_fn_improved import RewardImproved

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--reward_config', required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    r_cfg = load_config(args.reward_config)

    if 'threshold' in r_cfg:
        reward_fn = RewardUser(**r_cfg)
    else:
        reward_fn = RewardImproved(**r_cfg)

    # ここからEnv, Agentの初期化、学習ループへ…
    print("Loaded config:", cfg)
    print("Using reward function:", reward_fn.__class__.__name__)

if __name__ == '__main__':
    main()
