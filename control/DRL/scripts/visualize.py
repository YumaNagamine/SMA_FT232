#!/usr/bin/env python
import sys
import pandas as pd
import matplotlib.pyplot as plt

def visualize(log_csv):
    df = pd.read_csv(log_csv)
    plt.plot(df['step'], df['reward'])
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title('Training Curve')
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python visualize.py <log_csv>")
    else:
        visualize(sys.argv[1])
