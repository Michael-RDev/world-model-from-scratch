import argparse
import os
import json
import numpy as np
from pathlib import Path

from envs.car_env import CarDrivingEnv
from envs.balence_env import BalancingEnv

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "artifacts"


class OUActionNoise:
    def __init__(self, dim, theta=0.15, sigma=0.3, dt=1.0, rng=None):
        self.dim = dim
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.rng = rng or np.random.default_rng()
        self.state = np.zeros(dim, dtype=np.float32)

    def reset(self):
        self.state = np.zeros(self.dim, dtype=np.float32)

    def sample(self):
        noise = self.rng.normal(size=self.dim).astype(np.float32)
        self.state += (
            -self.theta * self.state * self.dt + self.sigma * np.sqrt(self.dt) * noise
        )
        return self.state.copy()


def collect_episode(env, max_steps, rng):
    if max_steps < 1:
        raise ValueError("max_steps must be positive")

    episode_seed = int(rng.integers(0, 2**31 - 1))
    obs, _ = env.reset(seed=episode_seed)
    noise = OUActionNoise(env.action_dim, rng=rng)

    observations = [obs.copy()]
    states = [env.get_state().copy()]
    actions, rewards, terminateds, truncateds = [], [], [], []

    for t in range(max_steps):
        action = env.clip_action(noise.sample())
        obs, reward, terminated, truncated, _ = env.step(action)

        # record if truncatd
        truncated = bool(truncated or (t + 1 == max_steps and not terminated))

        observations.append(obs.copy())
        states.append(env.get_state().copy())
        actions.append(action.copy())
        rewards.append(reward)
        terminateds.append(terminated)
        truncateds.append(truncated)

        if terminated or truncated:
            break

    obs_dtype = np.uint8 if env.obj_type == "image" else np.float32

    return {
        "observations": np.asarray(observations, dtype=obs_dtype),
        "states": np.asarray(states, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "terminated": np.asarray(terminateds, dtype=bool),
        "truncated": np.asarray(truncateds, dtype=bool),
        "episode_seed": np.int64(episode_seed),
    }


def collect_dataset(env_name, episodes, steps, obs_type, image_size, out_dir, seed=0):
    if episodes < 1 or steps < 1:
        raise ValueError("episodes and steps must be positive")

    env_classes = {"car": CarDrivingEnv, "balance": BalancingEnv}
    env = env_classes[env_name](
        obs_type=obs_type,
        image_size=image_size,
        max_steps=steps,
        seed=seed,
    )

    try:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=False)

        parameter_names = (
            "dt",
            "max_steps",
            "gravity",
            "mass_cart",
            "mass_pole",
            "pole_half_length",
            "force_mag",
            "theta_threshold",
            "x_threshold",
            "wheelbase",
            "max_steer",
            "max_accel",
            "max_speed",
            "min_speed",
            "friction",
            "road_width",
            "base_radius",
            "n_track_points",
            "world_half_extent",
            "randomize_track",
        )
        parameters = {
            name: np.asarray(getattr(env, name)).item()
            for name in parameter_names
            if hasattr(env, name)
        }
        metadata = {
            "schema_version": 1,
            "environment": env_name,
            "seed": seed,
            "episodes": episodes,
            "obs_type": obs_type,
            "image_size": list(image_size),
            "state_dim": env.state_dim,
            "action_dim": env.action_dim,
            "parameters": parameters,
            "numpy_version": np.__version__,
            "policy": {
                "name": "OUActionNoise",
                "theta": 0.15,
                "sigma": 0.3,
                "dt": 1.0,
            },
        }
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

        rng = np.random.default_rng(seed)
        lengths = []

        for episode in range(episodes):
            data = collect_episode(env, steps, rng)
            np.savez_compressed(out_dir / f"ep_{episode:04d}.npz", **data)
            lengths.append(len(data["actions"]))

        print(
            f"Saved {episodes} episodes to {out_dir}; "
            f"mean length: {np.mean(lengths):.1f} steps"
        )
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--env", choices=["car", "balance", "both"], default="both")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--obs-type", choices=["image", "state"], default="image")
    parser.add_argument("--image-size", type=int, nargs=2, default=[64, 64])
    parser.add_argument("--out", type=str, default=str(DATA_PATH))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    envs_to_run = ["car", "balance"] if args.env == "both" else [args.env]
    for name in envs_to_run:
        out_dir = os.path.join(args.out, name) if args.env == "both" else args.out
        collect_dataset(
            name,
            args.episodes,
            args.steps,
            args.obs_type,
            tuple(args.image_size),
            out_dir,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
