"""
Environment wrapper for ManiSkill environments with state observations.

Converts ManiSkill environment to DPPO-compatible format with state observations.
Modified from robomimic_lowdim.py to work with ManiSkill's gym interface.
"""

import numpy as np
import gym
from gym import spaces
import torch


class ManiskillLowdimWrapper(gym.Env):
    def __init__(
        self,
        env,
        normalization_path=None,
        low_dim_keys=None,  # ManiSkill uses different observation structure
        clamp_obs=False,
        init_state=None,
        render_hw=(256, 256),
        action_scale=1.0,
        policy="widowx_bridge",
    ):
        """
        Args:
            env: ManiSkill environment (gym.Env)
            normalization_path: Path to npz file with obs_min, obs_max, action_min, action_max
            low_dim_keys: List of observation keys to extract (for future use)
            clamp_obs: Whether to clamp observations to [-1, 1]
            init_state: Initial state for reset (not used for ManiSkill)
            render_hw: Render height and width
            action_scale: Scaling factor for actions
            policy: Policy type ("widowx_bridge" or "google_robot")
        """
        self.env = env
        self.init_state = init_state
        self.render_hw = render_hw
        self.clamp_obs = clamp_obs
        self.action_scale = action_scale
        self.policy = policy

        # set up normalization
        self.normalize = normalization_path is not None
        if self.normalize:
            normalization = np.load(normalization_path)
            self.obs_min = normalization["obs_min"]
            self.obs_max = normalization["obs_max"]
            self.action_min = normalization["action_min"]
            self.action_max = normalization["action_max"]

        # Get observation space from env
        # ManiSkill returns dict with "images" and "task_descriptions"
        # We need to extract state from raw observations
        raw_obs, _ = self.env.reset()
        
        # Extract state dimension from raw observation
        # For ManiSkill, we need to get state from the underlying env
        # This is a placeholder - actual state extraction depends on ManiSkill version
        if hasattr(self.env.env, 'unwrapped') and hasattr(self.env.env.unwrapped, 'get_obs'):
            # Try to get state observation
            state_obs = self._extract_state(raw_obs)
        else:
            # Fallback: use a default state dimension
            # This should be configured based on actual ManiSkill environment
            state_obs = np.zeros(23, dtype=np.float32)  # Default, should be set correctly
        
        # setup action space
        # ManiSkill actions are typically 7D: [world_vector(3), rotation_delta(3), gripper(1)]
        action_dim = 7
        low = np.full(action_dim, fill_value=-1)
        high = np.full(action_dim, fill_value=1)
        self.action_space = gym.spaces.Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=np.float32,
        )
        
        # setup observation space
        self.observation_space = spaces.Dict()
        low_obs = np.full_like(state_obs, fill_value=-1)
        high_obs = np.full_like(state_obs, fill_value=1)
        self.observation_space["state"] = spaces.Box(
            low=low_obs,
            high=high_obs,
            shape=low_obs.shape,
            dtype=np.float32,
        )

    def _extract_state(self, raw_obs):
        """
        Extract state observation from ManiSkill raw observation.
        This is a placeholder - actual implementation depends on ManiSkill version.
        """
        # ManiSkill typically returns dict with images and task_descriptions
        # We need to extract the actual state from the underlying environment
        # For now, return a placeholder - this should be implemented based on actual ManiSkill API
        if isinstance(raw_obs, dict):
            # Try to get state from underlying env
            if hasattr(self.env.env, 'unwrapped'):
                env_unwrapped = self.env.env.unwrapped
                # ManiSkill v2 may have different structure
                # This needs to be adapted based on actual ManiSkill version
                if hasattr(env_unwrapped, 'get_obs'):
                    return env_unwrapped.get_obs()
                # Try accessing state directly if available
                if hasattr(env_unwrapped, 'agent'):
                    # Extract state from agent
                    agent = env_unwrapped.agent
                    # Combine robot state (position, orientation, etc.)
                    # This is environment-specific and should be configured
                    return np.zeros(23, dtype=np.float32)  # Placeholder
        return np.zeros(23, dtype=np.float32)  # Placeholder

    def normalize_obs(self, obs):
        obs = 2 * (
            (obs - self.obs_min) / (self.obs_max - self.obs_min + 1e-6) - 0.5
        )  # -> [-1, 1]
        if self.clamp_obs:
            obs = np.clip(obs, -1, 1)
        return obs

    def unnormalize_action(self, action):
        action = (action + 1) / 2  # [-1, 1] -> [0, 1]
        return action * (self.action_max - self.action_min) + self.action_min

    def _prepare_action(self, action):
        """
        Convert DPPO action format to ManiSkill action format.
        Based on prepare_actions_for_maniskill from RLinf.
        """
        # action is in [-1, 1] format, shape (action_dim,)
        # Convert to ManiSkill format: world_vector, rotation_delta, gripper
        world_vector = action[:3] * self.action_scale
        rotation_delta = action[3:6] * self.action_scale
        open_gripper = action[6:7]  # range [0, 1] in normalized space
        
        # Convert to [0, 1] first
        open_gripper = (open_gripper + 1) / 2
        
        if self.policy == "widowx_bridge":
            gripper = 2.0 * (open_gripper > 0.5) - 1.0  # Binary: -1 or 1
        elif self.policy == "google_robot":
            raise NotImplementedError("Google robot policy not implemented")
        else:
            gripper = 2.0 * (open_gripper > 0.5) - 1.0
        
        # ManiSkill expects actions as numpy array or torch tensor
        # Format: [world_vector(3), rotation_delta(3), gripper(1)]
        maniskill_action = np.concatenate([world_vector, rotation_delta, gripper])
        return maniskill_action

    def get_observation(self, raw_obs):
        obs = {"state": self._extract_state(raw_obs)}
        if self.normalize:
            obs["state"] = self.normalize_obs(obs["state"])
        return obs

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed=seed)
            torch.manual_seed(seed)
        else:
            np.random.seed()
            torch.manual_seed(torch.randint(0, 2**32 - 1, (1,)).item())

    def reset(self, seed=None, options={}, **kwargs):
        """Reset the environment"""
        if seed is not None:
            self.seed(seed)
        
        raw_obs, info = self.env.reset(seed=seed, options=options)
        return self.get_observation(raw_obs), info

    def step(self, action):
        # Convert action format
        if self.normalize:
            action = self.unnormalize_action(action)
        
        maniskill_action = self._prepare_action(action)
        
        # Step environment
        raw_obs, reward, terminated, truncated, info = self.env.step(maniskill_action)
        obs = self.get_observation(raw_obs)
        
        # Convert terminated/truncated to boolean if needed
        if isinstance(terminated, torch.Tensor):
            terminated = terminated.cpu().item() if terminated.numel() == 1 else terminated.cpu().numpy()
        if isinstance(truncated, torch.Tensor):
            truncated = truncated.cpu().item() if truncated.numel() == 1 else truncated.cpu().numpy()
        
        # Return done=False for consistency with DPPO (episodes don't terminate early)
        return obs, reward, False, info

    def render(self, mode="rgb_array"):
        """Render the environment"""
        if hasattr(self.env, 'render'):
            return self.env.render()
        return None

    def close(self):
        if hasattr(self.env, 'close'):
            self.env.close()

