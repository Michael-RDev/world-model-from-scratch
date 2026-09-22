import numpy as np
import pygame
from .base_env import ENV2D


class BalancingEnv(ENV2D):
    def __init__(
        self,
        obs_type: str = "image",
        image_size=(64, 64),
        max_steps: int = 500,
        seed=None,
    ):
        super().__init__(
            obj_type=obs_type, image_size=image_size, max_steps=max_steps, seed=seed
        )

        # action: force in [-1, 1], scaled to +-force_mag newtons
        self.action_dim = 1
        self.action_low = np.array([-1.0], dtype=np.float32)
        self.action_high = np.array([1.0], dtype=np.float32)
        self.force_mag = 12.0

        self.image_size = image_size

        self.gravity = 9.8
        self.mass_cart = 1.0
        self.mass_pole = 0.1
        self.total_mass = self.mass_cart + self.mass_pole
        self.pole_half_length = 0.5
        self.dt = 0.02

        self.x_threshold = 3.0
        self.theta_threshold = np.deg2rad(
            75.0
        )  # Maximum tilt from upright before reset.

        self.world_half_extent = 3.6
        self._surface = pygame.Surface(self.image_size)

        self.state_dim = 5  # x, x_dot, cos(theta), sin(theta), theta_dot

        self.reset(seed=seed)

    def reset(self, seed=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self.x = float(self.np_random.uniform(-0.2, 0.2))
        self.x_dot = float(self.np_random.uniform(-0.05, 0.05))
        self.theta = float(self.np_random.uniform(-0.1, 0.1))
        self.theta_dot = float(self.np_random.uniform(-0.05, 0.05))
        self.step_count = 0
        return self.get_obs(), {}

    def step(self, action):
        action = self.clip_action(action)
        force = float(action[0]) * self.force_mag

        costheta, sintheta = np.cos(self.theta), np.sin(self.theta)
        temp = (
            force
            + self.mass_pole * self.pole_half_length * self.theta_dot**2 * sintheta
        ) / self.total_mass
        theta_acc = (self.gravity * sintheta - costheta * temp) / (
            self.pole_half_length
            * (4.0 / 3.0 - self.mass_pole * costheta**2 / self.total_mass)
        )
        x_acc = (
            temp
            - self.mass_pole
            * self.pole_half_length
            * theta_acc
            * costheta
            / self.total_mass
        )

        self.x += self.dt * self.x_dot
        self.x_dot += self.dt * x_acc
        self.theta += self.dt * self.theta_dot
        self.theta_dot += self.dt * theta_acc

        terminated = bool(
            abs(self.x) > self.x_threshold or abs(self.theta) > self.theta_threshold
        )
        self.step_count += 1
        truncated = self.step_count >= self.max_steps

        reward = float(np.cos(self.theta) - 0.05 * self.x**2 - 0.001 * force**2)
        if terminated:
            reward -= 5.0

        obs = self.get_obs()
        info = {"x": self.x, "theta": self.theta}
        return obs, reward, terminated, truncated, info

    def get_obs(self):
        return self.render() if self.obj_type == "image" else self.get_state()

    def get_state(self):
        return np.array(
            [
                self.x / self.x_threshold,
                self.x_dot / 5.0,
                np.cos(self.theta),
                np.sin(self.theta),
                self.theta_dot / 5.0,
            ],
            dtype=np.float32,
        )

    def w2s(self, x, y):
        w, h = self.image_size
        scale = w / (2 * self.world_half_extent)
        return w / 2 + x * scale, h * 0.75 - y * scale

    def render(self, *, as_surface=False):
        w, h = self.image_size
        ss = 2
        scale = w * ss / (2 * self.world_half_extent)
        cx, cy = self.w2s(self.x, 0.0)
        cx, cy = round(cx * ss), round(cy * ss)
        pole_len_px = 2 * self.pole_half_length * scale
        rail_y = cy + round(0.25 * scale)
        cache_key = (w, h, self.world_half_extent, self.x_threshold)
        cached = getattr(self, "_render_background", None)
        if cached is None or cached[0] != cache_key:
            surf = pygame.Surface((w * ss, h * ss))
            for row in range(h * ss):
                shade = int(9 * row / max(1, h * ss - 1))
                pygame.draw.line(
                    surf, (18 + shade, 27 + shade, 41 + shade), (0, row), (w * ss, row)
                )
            grid = max(8, round(0.5 * scale))
            for x in range(w * ss // 2 % grid, w * ss, grid):
                pygame.draw.line(surf, (34, 46, 62), (x, 0), (x, h * ss))
            for y in range(round(h * ss * 0.75) % grid, h * ss, grid):
                pygame.draw.line(surf, (34, 46, 62), (0, y), (w * ss, y))

            # Ground rail, graduations, and end stops make motion easy to read.
            pygame.draw.rect(surf, (13, 21, 32), (0, rail_y, w * ss, h * ss - rail_y))
            for x in np.arange(-3.5, 3.6, 0.25):
                px = round(self.w2s(x, 0)[0] * ss)
                major = abs(x - round(x)) < 1e-6
                pygame.draw.line(
                    surf,
                    (97, 118, 138) if major else (51, 70, 88),
                    (px, rail_y + 14 * ss),
                    (px, rail_y + (25 if major else 20) * ss),
                    max(1, ss),
                )
            for dy, color, thickness in [
                (6, (8, 15, 24), 9),
                (0, (96, 119, 138), 5),
                (-3, (170, 189, 199), 1),
            ]:
                pygame.draw.line(
                    surf,
                    color,
                    (0, rail_y + dy * ss),
                    (w * ss, rail_y + dy * ss),
                    thickness * ss,
                )
            for side in (-1, 1):
                px = round(self.w2s(side * self.x_threshold, 0)[0] * ss)
                pygame.draw.rect(
                    surf,
                    (54, 69, 84),
                    (px - 5 * ss, rail_y - 15 * ss, 10 * ss, 25 * ss),
                    border_radius=2 * ss,
                )
                pygame.draw.rect(
                    surf,
                    (232, 166, 92),
                    (px - 3 * ss, rail_y - 13 * ss, 6 * ss, 9 * ss),
                    border_radius=ss,
                )
            self._render_background = (cache_key, surf)
        surf = self._render_background[1].copy()

        # Upright reference and angular sweep are visual guides only.
        for y in range(round(cy - pole_len_px - 20 * ss), cy, 12 * ss):
            pygame.draw.line(
                surf, (63, 92, 104), (cx, y), (cx, min(y + 5 * ss, cy)), ss
            )
        for angle in (-self.theta_threshold, self.theta_threshold):
            end = (
                round(cx + pole_len_px * np.sin(angle)),
                round(cy - pole_len_px * np.cos(angle)),
            )
            pygame.draw.line(surf, (46, 57, 71), (cx, cy), end, ss)
        arc = [
            (round(cx + 0.4 * scale * np.sin(a)), round(cy - 0.4 * scale * np.cos(a)))
            for a in np.linspace(0, self.theta, 24)
        ]
        pygame.draw.lines(surf, (234, 175, 99), False, arc, max(1, 2 * ss))

        cart_w, cart_h = round(0.74 * scale), round(0.32 * scale)
        cart = pygame.Rect(0, 0, cart_w, cart_h)
        cart.center = (cx, cy)
        shadow = cart.move(3 * ss, 6 * ss)
        pygame.draw.rect(surf, (8, 15, 23), shadow, border_radius=5 * ss)
        wheel_r = max(2, round(0.085 * scale))
        for side in (-1, 1):
            wheel = (cx + round(side * 0.24 * scale), rail_y - wheel_r - ss)
            pygame.draw.circle(surf, (10, 17, 25), wheel, wheel_r + 2 * ss)
            pygame.draw.circle(surf, (105, 130, 151), wheel, wheel_r)
            pygame.draw.circle(surf, (31, 47, 65), wheel, max(1, wheel_r // 2))
        pygame.draw.rect(surf, (64, 130, 194), cart, border_radius=5 * ss)
        top = pygame.Rect(cart.left, cart.top, cart.width, max(1, cart.height // 3))
        pygame.draw.rect(surf, (120, 191, 237), top, border_radius=3 * ss)
        for side in (-1, 1):
            bolt = (cx + round(side * 0.26 * scale), cy + round(0.035 * scale))
            pygame.draw.circle(surf, (28, 71, 112), bolt, max(1, 2 * ss))

        tip = (
            round(cx + pole_len_px * np.sin(self.theta)),
            round(cy - pole_len_px * np.cos(self.theta)),
        )
        rod_width = max(2, round(0.065 * scale))
        pygame.draw.line(
            surf,
            (8, 16, 25),
            (cx + 3 * ss, cy + 3 * ss),
            (tip[0] + 3 * ss, tip[1] + 3 * ss),
            rod_width + 2 * ss,
        )
        pygame.draw.line(surf, (218, 167, 101), (cx, cy), tip, rod_width)
        pygame.draw.line(
            surf,
            (255, 224, 164),
            (cx - ss, cy),
            (tip[0] - ss, tip[1]),
            max(1, rod_width // 3),
        )
        bob_r = max(3, round(0.12 * scale))
        pygame.draw.circle(surf, (89, 61, 40), tip, bob_r + ss)
        pygame.draw.circle(surf, (242, 182, 104), tip, bob_r)
        pygame.draw.circle(
            surf,
            (255, 224, 170),
            (tip[0] - bob_r // 3, tip[1] - bob_r // 3),
            max(1, bob_r // 3),
        )
        pivot_r = max(2, round(0.08 * scale))
        pygame.draw.circle(surf, (186, 208, 222), (cx, cy), pivot_r)
        pygame.draw.circle(surf, (33, 63, 86), (cx, cy), max(1, pivot_r // 2))

        pygame.transform.smoothscale(surf, self.image_size, self._surface)
        if as_surface:
            return self._surface
        arr = pygame.surfarray.array3d(self._surface)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def close(self):
        pass
