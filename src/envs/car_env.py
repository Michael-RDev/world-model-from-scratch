import numpy as np
import pygame

from .base_env import ENV2D


class CarDrivingEnv(ENV2D):
    def __init__(
        self,
        obs_type: str = "image",
        image_size=(64, 64),
        max_steps: int = 500,
        randomize_track: bool = True,
        seed=None,
    ):
        super().__init__(
            obj_type=obs_type, image_size=image_size, max_steps=max_steps, seed=seed
        )

        # [steer, throttle], both in [-1, 1]
        self.action_dim = 2
        self.action_low = np.array([-1.0, -1.0], dtype=np.float32)
        self.action_high = np.array([1.0, 1.0], dtype=np.float32)

        self.dt = 0.1
        self.wheelbase = 2.5
        self.max_steer = 0.5
        self.max_accel = 6.0
        self.max_speed = 12.0
        self.min_speed = -4.0
        self.friction = 0.98
        self.image_size = image_size

        self.randomize_track = randomize_track
        self.road_width = 4.0
        self.base_radius = 15.0
        self.n_track_points = 200

        self.world_half_extent = 24.0  # world coords span [-extent, extent]
        self.surface = pygame.Surface(image_size)

        self.state_dim = 7  # x, y, cos(h), sin(h), v, lateral_off, heading_err

        self.build_track(seed)
        self.reset(seed=seed)

    def build_track(self, seed=None):
        rng = np.random.default_rng(seed) if seed is not None else self.np_random
        thetas = np.linspace(0, 2 * np.pi, self.n_track_points, endpoint=False)

        if self.randomize_track:
            radius = np.full_like(thetas, self.base_radius)
            for k in range(1, 5):
                amp = rng.uniform(0.5, 2.5) / k
                phase = rng.uniform(0, 2 * np.pi)
                radius += amp * np.sin(k * thetas + phase)
            radius = np.clip(radius, self.base_radius * 0.5, self.base_radius * 1.4)
        else:
            radius = np.full_like(thetas, self.base_radius)

        self.track_x = radius * np.cos(thetas)
        self.track_y = radius * np.sin(thetas)
        self.track_pts = np.stack([self.track_x, self.track_y], axis=1)  # (N, 2)

        seg = np.roll(self.track_pts, -1, axis=0) - self.track_pts
        seg_len = np.linalg.norm(seg, axis=1)
        seg_len = np.clip(seg_len, 1e-6, None)
        self.track_tangent = seg / seg_len[:, None]

    def _nearest_track_index(self, pos):
        d = self.track_pts - pos[None, :]
        dist2 = np.einsum("ij,ij->i", d, d)
        idx = int(np.argmin(dist2))
        return idx, float(np.sqrt(dist2[idx]))

    def reset(self, seed=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        if self.randomize_track:
            self.build_track(seed=int(self.np_random.integers(0, 2**31 - 1)))

        start_idx = 0
        self.x, self.y = self.track_pts[start_idx]
        tangent = self.track_tangent[start_idx]
        self.heading = float(np.arctan2(tangent[1], tangent[0]))
        self.speed = 0.0
        self._prev_idx = start_idx
        self._step_count = 0

        obs = self.get_obs()
        info = {"track_index": start_idx}
        return obs, info

    def step(self, action):
        action = self.clip_action(action)
        steer, throttle = float(action[0]), float(action[1])
        steer *= self.max_steer
        accel = throttle * self.max_accel

        # kinematic bicycle model
        self.speed = np.clip(
            self.speed + accel * self.dt, self.min_speed, self.max_speed
        )
        self.speed *= self.friction
        self.x += self.speed * np.cos(self.heading) * self.dt
        self.y += self.speed * np.sin(self.heading) * self.dt
        self.heading += (self.speed / self.wheelbase) * np.tan(steer) * self.dt
        self.heading = (self.heading + np.pi) % (2 * np.pi) - np.pi

        idx, lateral_off = self._nearest_track_index(np.array([self.x, self.y]))

        # progress along the track (handles wrap-around at the start/finish)
        n = self.n_track_points
        forward = (idx - self._prev_idx) % n
        backward = (self._prev_idx - idx) % n
        step_progress = forward if forward <= backward else -backward
        self._prev_idx = idx

        off_track = lateral_off > self.road_width
        reward = 0.1 * step_progress - 0.05 * lateral_off - 0.001 * (steer**2)
        if off_track:
            reward -= 5.0

        self._step_count += 1
        terminated = bool(off_track)
        truncated = self._step_count >= self.max_steps

        obs = self.get_obs()
        info = {"track_index": idx, "lateral_offset": lateral_off, "speed": self.speed}
        return obs, reward, terminated, truncated, info

    def get_obs(self):
        return self.render() if self.obj_type == "image" else self.get_state()

    def get_state(self):
        idx, lateral_off = self._nearest_track_index(np.array([self.x, self.y]))
        tangent = self.track_tangent[idx]
        track_heading = np.arctan2(tangent[1], tangent[0])
        heading_err = (self.heading - track_heading + np.pi) % (2 * np.pi) - np.pi
        return np.array(
            [
                self.x / self.world_half_extent,
                self.y / self.world_half_extent,
                np.cos(self.heading),
                np.sin(self.heading),
                self.speed / self.max_speed,
                lateral_off / self.road_width,
                heading_err,
            ],
            dtype=np.float32,
        )

    def w2s(self, x, y):
        w, h = self.image_size
        scale = min(w, h) / (2 * self.world_half_extent)
        return w / 2 + x * scale, h / 2 - y * scale

    def render(self, *, as_surface=False):
        w, h = self.image_size
        ss = 2
        scale = min(w, h) * ss / (2 * self.world_half_extent)

        def point(x, y):
            px, py = self.w2s(x, y)
            return round(px * ss), round(py * ss)

        cache_key = (
            w,
            h,
            self.world_half_extent,
            self.road_width,
            self.track_pts.tobytes(),
        )
        cached = getattr(self, "_render_background", None)
        if cached is None or cached[0] != cache_key:
            surf = pygame.Surface((w * ss, h * ss))
            for row in range(h * ss):
                shade = int(7 * row / max(1, h * ss - 1))
                pygame.draw.line(
                    surf, (22 + shade, 43 + shade, 40 + shade), (0, row), (w * ss, row)
                )
            # A restrained field pattern gives the circuit depth without randomness.
            grid = max(8, round(3 * scale))
            for x in range(0, w * ss, grid):
                pygame.draw.line(surf, (35, 57, 51), (x, 0), (x, h * ss))
            for y in range(0, h * ss, grid):
                pygame.draw.line(surf, (35, 57, 51), (0, y), (w * ss, y))

            tangents = np.roll(self.track_pts, -1, axis=0) - np.roll(
                self.track_pts, 1, axis=0
            )
            tangents /= np.maximum(
                np.linalg.norm(tangents, axis=1, keepdims=True), 1e-6
            )
            normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
            pts = [point(x, y) for x, y in self.track_pts]

            def ribbon(half_width, color, offset=(0, 0)):
                outer = [point(*p) for p in self.track_pts + half_width * normals]
                inner = [point(*p) for p in self.track_pts - half_width * normals]
                outline = outer + [outer[0]] + list(reversed(inner + [inner[0]]))
                pygame.draw.polygon(
                    surf, color, [(x + offset[0], y + offset[1]) for x, y in outline]
                )

            ribbon(self.road_width + 0.3, (14, 28, 28), (2 * ss, 4 * ss))
            ribbon(self.road_width + 0.12, (99, 112, 109))
            ribbon(self.road_width, (43, 50, 59))
            ribbon(self.road_width - 0.2, (48, 56, 65))

            # Curbs follow both edges; the asphalt spans the existing off-track limit.
            for side in (-1, 1):
                edge = self.track_pts + side * self.road_width * normals
                edge_pts = [point(x, y) for x, y in edge]
                for i in range(len(edge_pts)):
                    color = (218, 221, 207) if (i // 4) % 2 else (195, 92, 77)
                    pygame.draw.line(
                        surf,
                        color,
                        edge_pts[i],
                        edge_pts[(i + 1) % len(pts)],
                        max(1, round(0.28 * scale)),
                    )

            for i in range(0, len(pts), 10):
                dash = [pts[j % len(pts)] for j in range(i, i + 4)]
                pygame.draw.lines(
                    surf, (167, 178, 172), False, dash, max(1, round(1.5 * ss))
                )

            # Start/finish tiles are oriented across the track, not the screen.
            origin, tangent, normal = self.track_pts[0], tangents[0], normals[0]
            tile_count = 18
            tile = 2 * self.road_width / tile_count
            for row in range(2):
                for col in range(tile_count):
                    across = -self.road_width + col * tile
                    forward = (row - 1) * tile
                    corners = [
                        origin + tangent * u + normal * v
                        for u, v in (
                            (forward, across),
                            (forward + tile, across),
                            (forward + tile, across + tile),
                            (forward, across + tile),
                        )
                    ]
                    color = (225, 231, 221) if (row + col) % 2 else (24, 31, 38)
                    pygame.draw.polygon(surf, color, [point(*p) for p in corners])

            self._render_background = (cache_key, surf)
        surf = self._render_background[1].copy()

        # Vehicle geometry stays in the car's local frame as it rotates.
        c, s = np.cos(self.heading), np.sin(self.heading)

        def body(points, color, offset=(0, 0)):
            vertices = [
                point(self.x + lx * c - ly * s, self.y + lx * s + ly * c)
                for lx, ly in points
            ]
            vertices = [(px + offset[0], py + offset[1]) for px, py in vertices]
            pygame.draw.polygon(surf, color, vertices)

        silhouette = [
            (-0.94, -0.47),
            (0.62, -0.47),
            (0.98, -0.31),
            (0.98, 0.31),
            (0.62, 0.47),
            (-0.94, 0.47),
        ]
        body(silhouette, (12, 22, 28), (ss, 2 * ss))
        for axle in (-0.57, 0.53):
            for side in (-1, 1):
                y = side * 0.49
                body(
                    [
                        (axle - 0.22, y - 0.12),
                        (axle + 0.22, y - 0.12),
                        (axle + 0.22, y + 0.12),
                        (axle - 0.22, y + 0.12),
                    ],
                    (13, 20, 27),
                )
        body(silhouette, (69, 208, 179))
        body(
            [
                (-0.65, -0.36),
                (0.52, -0.36),
                (0.72, -0.26),
                (0.72, 0.26),
                (0.52, 0.36),
                (-0.65, 0.36),
            ],
            (109, 230, 200),
        )
        body([(-0.48, -0.29), (0.30, -0.29), (0.42, 0.29), (-0.48, 0.29)], (25, 51, 64))
        body(
            [(-0.30, -0.25), (0.10, -0.25), (0.10, 0.25), (-0.30, 0.25)], (66, 159, 153)
        )
        for side in (-1, 1):
            y = side * 0.31
            body(
                [
                    (0.76, y - 0.075),
                    (0.94, y - 0.075),
                    (0.94, y + 0.075),
                    (0.76, y + 0.075),
                ],
                (244, 248, 222),
            )
            body(
                [
                    (-0.95, y - 0.09),
                    (-0.80, y - 0.09),
                    (-0.80, y + 0.09),
                    (-0.95, y + 0.09),
                ],
                (242, 103, 89),
            )

        pygame.transform.smoothscale(surf, self.image_size, self.surface)
        if as_surface:
            return self.surface
        arr = pygame.surfarray.array3d(self.surface)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def close(self):
        pass
