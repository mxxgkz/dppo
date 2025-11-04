"""
Process ManiSkill dataset and save it into DPPO custom format (npz) so it can be loaded for diffusion training.

This script processes ManiSkill demonstrations. ManiSkill datasets can be in various formats:
1. HDF5 files (similar to Robomimic)
2. Replay buffers
3. Trajectory files

This is a template that may need adaptation based on the specific ManiSkill dataset format.
"""

import os
import numpy as np
from tqdm import tqdm
import random
import logging
from copy import deepcopy
import h5py


def make_dataset(
    load_path,
    save_dir,
    save_name_prefix,
    val_split,
    normalize=True,
    obs_keys=None,
):
    """
    Process ManiSkill dataset from load_path and save to npz format.
    
    Args:
        load_path: Path to ManiSkill dataset file (HDF5 or other format)
        save_dir: Directory to save processed dataset
        save_name_prefix: Prefix for saved files
        val_split: Validation split ratio (0-1)
        normalize: Whether to normalize observations and actions
        obs_keys: List of observation keys to extract (if None, uses default)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Determine file format
    if load_path.endswith('.hdf5') or load_path.endswith('.h5'):
        # Process HDF5 format (similar to Robomimic)
        process_hdf5_dataset(
            load_path, save_dir, save_name_prefix, val_split, normalize, obs_keys
        )
    else:
        raise ValueError(f"Unsupported file format: {load_path}")


def process_hdf5_dataset(
    load_path, save_dir, save_name_prefix, val_split, normalize, obs_keys
):
    """Process HDF5 format ManiSkill dataset."""
    logging.info(f"Loading dataset from {load_path}")
    
    with h5py.File(load_path, 'r') as f:
        # Get list of demonstrations
        demos = list(f['data'].keys())
        logging.info(f"Found {len(demos)} demonstrations")
        
        # Extract observation and action dimensions from first demo
        first_demo = demos[0]
        if obs_keys is None:
            # Default observation keys for ManiSkill
            # These may need to be adjusted based on actual ManiSkill environment
            obs_keys = list(f[f'data/{first_demo}/obs'].keys())
            # Filter to only low-dim observations (exclude images)
            obs_keys = [k for k in obs_keys if 'image' not in k.lower() and 'rgb' not in k.lower()]
        
        # Get dimensions
        sample_obs = f[f'data/{first_demo}/obs/{obs_keys[0]}']
        obs_dim = sample_obs.shape[-1] if len(sample_obs.shape) > 1 else 1
        if len(obs_keys) > 1:
            # Sum dimensions from all keys
            obs_dim = sum([f[f'data/{first_demo}/obs/{k}'].shape[-1] if len(f[f'data/{first_demo}/obs/{k}'].shape) > 1 else 1 
                          for k in obs_keys])
        
        sample_action = f[f'data/{first_demo}/actions']
        action_dim = sample_action.shape[-1]
        
        logging.info(f"Observation dimension: {obs_dim}, Action dimension: {action_dim}")
        logging.info(f"Observation keys: {obs_keys}")
        
        # Calculate normalization stats if needed
        if normalize:
            obs_all = []
            action_all = []
            for ep in tqdm(demos, desc="Computing normalization stats"):
                traj_length = f[f'data/{ep}'].attrs.get('num_samples', len(f[f'data/{ep}/actions']))
                
                # Concatenate observations
                obs_list = [f[f'data/{ep}/obs/{key}'][()] for key in obs_keys]
                obs = np.hstack(obs_list) if len(obs_list) > 1 else obs_list[0]
                
                actions = f[f'data/{ep}/actions'][()]
                obs_all.append(obs)
                action_all.append(actions)
            
            obs_all = np.concatenate(obs_all, axis=0)
            action_all = np.concatenate(action_all, axis=0)
            
            obs_min = np.min(obs_all, axis=0)
            obs_max = np.max(obs_all, axis=0)
            action_min = np.min(action_all, axis=0)
            action_max = np.max(action_all, axis=0)
            
            logging.info(f"Obs min: {obs_min}, Obs max: {obs_max}")
            logging.info(f"Action min: {action_min}, Action max: {action_max}")
        else:
            obs_min = None
            obs_max = None
            action_min = None
            action_max = None
        
        # Split into train and validation
        num_traj = len(demos)
        num_train = int(num_traj * (1 - val_split))
        train_indices = random.sample(range(num_traj), k=num_train)
        
        # Prepare data containers
        out_train = {
            "states": [],
            "actions": [],
            "rewards": [],
            "traj_lengths": [],
        }
        out_val = deepcopy(out_train)
        
        # Process each demo
        for i, ep in tqdm(enumerate(demos), total=len(demos), desc="Processing trajectories"):
            out = out_train if i in train_indices else out_val
            
            # Get trajectory data
            traj_length = f[f'data/{ep}'].attrs.get('num_samples', len(f[f'data/{ep}/actions']))
            out["traj_lengths"].append(traj_length)
            
            # Extract observations
            obs_list = [f[f'data/{ep}/obs/{key}'][()] for key in obs_keys]
            raw_obs = np.hstack(obs_list) if len(obs_list) > 1 else obs_list[0]
            
            raw_actions = f[f'data/{ep}/actions'][()]
            rewards = f[f'data/{ep}/rewards'][()] if 'rewards' in f[f'data/{ep}'] else np.zeros(traj_length)
            
            # Normalize if specified
            if normalize:
                obs = 2 * (raw_obs - obs_min) / (obs_max - obs_min + 1e-6) - 1
                actions = 2 * (raw_actions - action_min) / (action_max - action_min + 1e-6) - 1
            else:
                obs = raw_obs
                actions = raw_actions
            
            # Store trajectories
            out["states"].append(obs)
            out["actions"].append(actions)
            out["rewards"].append(rewards)
        
        # Concatenate trajectories
        for key in ["states", "actions", "rewards"]:
            out_train[key] = np.concatenate(out_train[key], axis=0)
            if val_split > 0:
                out_val[key] = np.concatenate(out_val[key], axis=0)
        
        # Save datasets
        train_save_path = os.path.join(save_dir, save_name_prefix + "train.npz")
        np.savez_compressed(
            train_save_path,
            states=np.array(out_train["states"]),
            actions=np.array(out_train["actions"]),
            rewards=np.array(out_train["rewards"]),
            terminals=np.array([False] * len(out_train["states"])),
            traj_lengths=np.array(out_train["traj_lengths"]),
        )
        logging.info(f"Saved training dataset to {train_save_path}")
        
        if val_split > 0:
            val_save_path = os.path.join(save_dir, save_name_prefix + "val.npz")
            np.savez_compressed(
                val_save_path,
                states=np.array(out_val["states"]),
                actions=np.array(out_val["actions"]),
                rewards=np.array(out_val["rewards"]),
                terminals=np.array([False] * len(out_val["states"])),
                traj_lengths=np.array(out_val["traj_lengths"]),
            )
            logging.info(f"Saved validation dataset to {val_save_path}")
        
        # Save normalization stats
        if normalize:
            normalization_save_path = os.path.join(
                save_dir, save_name_prefix + "normalization.npz"
            )
            np.savez_compressed(
                normalization_save_path,
                obs_min=obs_min,
                obs_max=obs_max,
                action_min=action_min,
                action_max=action_max,
            )
            logging.info(f"Saved normalization stats to {normalization_save_path}")
        
        # Logging summary
        logging.info("\n========== Final ===========")
        logging.info(
            f"Train - Trajectories: {len(out_train['traj_lengths'])}, Transitions: {np.sum(out_train['traj_lengths'])}"
        )
        if val_split > 0:
            logging.info(
                f"Val - Trajectories: {len(out_val['traj_lengths'])}, Transitions: {np.sum(out_val['traj_lengths'])}"
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process ManiSkill dataset for DPPO training"
    )
    parser.add_argument("--load_path", type=str, required=True, help="Path to ManiSkill dataset file")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save processed dataset")
    parser.add_argument("--save_name_prefix", type=str, default="maniskill_", help="Prefix for saved files")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio (0-1)")
    parser.add_argument("--normalize", action="store_true", help="Normalize observations and actions")
    parser.add_argument("--obs_keys", nargs="*", default=None, help="Observation keys to extract")
    args = parser.parse_args()

    import datetime

    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(
        args.save_dir,
        args.save_name_prefix
        + f"_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log",
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(),
        ],
    )

    make_dataset(
        args.load_path,
        args.save_dir,
        args.save_name_prefix,
        args.val_split,
        args.normalize,
        args.obs_keys,
    )

