import argparse
import os
import numpy as np

from envs.car_env import CarDrivingEnv
from envs.balence_env import BalancingEnv


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
    obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    noise = OUActionNoise(env.action_dim, rng=rng)

    observations = [obs]
    actions, rewards, terminateds = [], [], []

    for _ in range(max_steps):
        raw = noise.sample()
        action = np.clip(raw, env.action_low, env.action_high)
        obs, reward, terminated, truncated, info = env.step(action)

        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        terminateds.append(terminated)

        if terminated or truncated:
            break

    obs_dtype = np.float32 if env.obs_type == "state" else np.uint8
    return {
        "observations": np.stack(observations).astype(obs_dtype),
        "actions": np.stack(actions).astype(np.float32),
        "rewards": np.array(rewards, dtype=np.float32),
        "terminated": np.array(terminateds, dtype=bool),
    }


def collect_dataset(env_name, episodes, steps, obs_type, image_size, out_dir, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    if env_name == "car":
        env = CarDrivingEnv(obs_type=obs_type, image_size=image_size, max_steps=steps)
    elif env_name == "balance":
        env = BalancingEnv(obs_type=obs_type, image_size=image_size, max_steps=steps)
    else:
        raise ValueError(env_name)

    lengths = []
    for ep in range(episodes):
        data = collect_episode(env, steps, rng)
        lengths.append(len(data["actions"]))
        path = os.path.join(out_dir, f"ep_{ep:04d}.npz")
        np.savez_compressed(path, **data)

    env.close()
    print(
        f"[{env_name}] saved {episodes} episodes to {out_dir} "
        f"(mean length {np.mean(lengths):.1f} steps)"
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--env", choices=["car", "balance", "both"], default="both")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--obs-type", choices=["image", "state"], default="image")
    parser.add_argument("--image-size", type=int, nargs=2, default=[64, 64])
    parser.add_argument("--out", type=str, default="data")
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
