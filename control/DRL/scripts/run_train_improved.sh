#!/usr/bin/env bash

# SAC の基本設定
CONFIG="configs/sac.yaml"
# 改良案報酬関数用設定
REWARD_CONFIG="configs/reward_improved.yaml"

python src/train.py \
    --config "$CONFIG" \
    --reward_config "$REWARD_CONFIG"
