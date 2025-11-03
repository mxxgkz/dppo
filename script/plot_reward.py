#!/usr/bin/env python3
"""
Plot rewards from run.log file
"""

import re
import matplotlib.pyplot as plt
import argparse
import sys


def parse_log_file(log_path):
    """Parse run.log and extract training and eval rewards"""
    train_steps = []
    train_rewards = []
    eval_iterations = []
    eval_rewards = []
    
    with open(log_path, 'r') as f:
        for line in f:
            # Parse training reward lines like:
            # "1: step    80000 | ... | reward 1266.5931 | ..."
            train_match = re.search(r'(\d+):\s+step\s+(\d+)\s+\|\s+.*?\|\s+reward\s+([\d.]+)', line)
            if train_match:
                iteration = int(train_match.group(1))
                step = int(train_match.group(2))
                reward = float(train_match.group(3))
                train_steps.append(step)
                train_rewards.append(reward)
            
            # Parse eval reward lines like:
            # "eval: success rate   1.0000 | avg episode reward 1424.7304 | ..."
            eval_match = re.search(r'eval:.*?avg episode reward\s+([\d.]+)', line)
            if eval_match:
                reward = float(eval_match.group(1))
                # Use current training iteration (last seen) for eval
                eval_iterations.append(len(train_steps))
                eval_rewards.append(reward)
    
    return train_steps, train_rewards, eval_iterations, eval_rewards


def plot_rewards(log_path, save_path=None, show=True):
    """Plot training and evaluation rewards"""
    train_steps, train_rewards, eval_iterations, eval_rewards = parse_log_file(log_path)
    
    if not train_rewards and not eval_rewards:
        print(f"No reward data found in {log_path}")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot training rewards
    if train_steps and train_rewards:
        ax1.plot(train_steps, train_rewards, 'b-', label='Training Reward', linewidth=2)
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Reward')
        ax1.set_title('Training Rewards Over Time')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
    
    # Plot evaluation rewards
    if eval_iterations and eval_rewards:
        ax2.plot(eval_iterations, eval_rewards, 'r-o', label='Evaluation Reward', linewidth=2, markersize=6)
        ax2.set_xlabel('Training Iteration')
        ax2.set_ylabel('Reward')
        ax2.set_title('Evaluation Rewards Over Time')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot rewards from run.log')
    parser.add_argument('log_path', type=str, help='Path to run.log file')
    parser.add_argument('--save', '-s', type=str, default=None, 
                       help='Path to save the plot (e.g., reward_plot.png)')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display the plot (useful for saving only)')
    
    args = parser.parse_args()
    
    plot_rewards(args.log_path, save_path=args.save, show=not args.no_show)


if __name__ == '__main__':
    main()

