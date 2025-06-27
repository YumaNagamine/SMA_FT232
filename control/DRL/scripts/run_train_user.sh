
# SAC の基本設定
CONFIG="configs/sac.yaml"
# ユーザー提案報酬関数用設定
REWARD_CONFIG="configs/reward_user.yaml"

python src/train.py \
    --config "$CONFIG" \
    --reward_config "$REWARD_CONFIG"