import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.car_env import CarDrivingEnv
from envs.balence_env import BalancingEnv


RENDER_FPS = 60
PHYSICS_HZ = 120
BALANCE_KEY_FORCE_SCALE = 1.0
BALANCE_TIME_SCALE = 0.5


class KeyboardControls:
    def __init__(self):
        self.held = set()
        self.focused = True

    def clear(self):
        self.held.clear()

    def process(self, events):
        command = None
        for event in events:
            if event.type == pygame.QUIT:
                return "quit"
            if event.type in (pygame.WINDOWFOCUSLOST, pygame.WINDOWMINIMIZED):
                self.clear()
                self.focused = False
            elif event.type == pygame.WINDOWFOCUSGAINED:
                self.clear()
                self.focused = True
            elif event.type == pygame.KEYUP:
                self.held.discard(event.key)
            elif event.type == pygame.KEYDOWN and self.focused:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    return "quit"
                if not getattr(event, "repeat", False):
                    if event.key == pygame.K_TAB:
                        command = "switch"
                    elif event.key == pygame.K_r:
                        command = "reset"
                self.held.add(event.key)
        return command

    def action(self, is_car):
        if not self.focused or pygame.K_SPACE in self.held:
            return np.zeros(2 if is_car else 1, dtype=np.float32)
        left = bool(self.held & {pygame.K_a, pygame.K_LEFT})
        right = bool(self.held & {pygame.K_d, pygame.K_RIGHT})
        if is_car:
            forward = bool(self.held & {pygame.K_w, pygame.K_UP})
            reverse = bool(self.held & {pygame.K_s, pygame.K_DOWN})
            steering = 0 if pygame.K_e in self.held else int(left) - int(right)
            return np.array([steering, int(forward) - int(reverse)], dtype=np.float32)
        return np.array(
            [(int(right) - int(left)) * BALANCE_KEY_FORCE_SCALE], dtype=np.float32
        )


class FixedStepClock:
    def __init__(self, dt=1.0 / PHYSICS_HZ, *, time_scale=1.0):
        self.dt = dt
        self.time_scale = time_scale
        self.pending = 0.0

    def advance(self, elapsed):
        self.pending += min(max(elapsed, 0.0), 0.1) * self.time_scale
        count = int((self.pending + 1e-10) / self.dt)
        self.pending = max(0.0, self.pending - count * self.dt)
        return count


class DemoDisplay:
    def __init__(self, env):
        pygame.font.init()
        self.is_car = env.action_dim == 2
        w, h = env.image_size
        self.pad, self.header, self.footer = 16, 142, 110
        self.width = w + self.pad * 2
        self.height = h
        self.ink, self.muted = (225, 236, 239), (119, 137, 149)
        self.accent = (69, 208, 179) if self.is_car else (242, 182, 104)
        self.rule, self.key_bg = (35, 47, 57), (26, 35, 44)
        self.fonts = {
            size: pygame.font.Font(None, size) for size in (16, 18, 20, 28, 30)
        }
        self.base = pygame.Surface((self.width, h + self.header + self.footer))
        self.base.fill((15, 19, 24))
        self.frame = self.base.copy()
        self.hints = []
        self.text(self.base, "WORLD MODEL LAB", 24, 18, 18, self.accent)
        title = "01 / CIRCUIT DRIVING" if self.is_car else "02 / INVERTED PENDULUM"
        self.text(self.base, title, 24, 44, 30)
        pygame.draw.circle(self.base, self.accent, (self.width - 104, 23), 3)
        self.text(self.base, "MANUAL", self.width - 93, 16, 16, self.muted)
        pygame.draw.line(self.base, self.rule, (24, 79), (self.width - 24, 79))
        labels = (
            ("SPEED", "HEADING", "THROTTLE")
            if self.is_car
            else ("POLE ANGLE", "CART POSITION", "APPLIED FORCE")
        )
        self.cell_w = (self.width - 48) // 3
        for i, label in enumerate(labels):
            x = 24 + i * self.cell_w
            if i:
                pygame.draw.line(self.base, self.rule, (x - 14, 95), (x - 14, 125))
            self.text(self.base, label, x, 92, 16, self.muted)
        pygame.draw.rect(
            self.base, self.rule, (self.pad - 1, self.header - 1, w + 2, h + 2), 1
        )
        y = self.header + h + 16
        if self.is_car:
            x = self.add_hint("W/S", "Throttle", 24, y, "throttle")
            x = self.add_hint("A/D", "Steer", x, y, "steer")
            x = self.add_hint("E", "Straighten", x, y, "straight")
        else:
            x = self.add_hint("A/D", "Push cart", 24, y, "steer")
        self.add_hint("SPACE", "Neutral", x, y, "neutral")
        x = self.add_hint("R", "Reset", 24, y + 31)
        x = self.add_hint("TAB", "Switch environment", x, y + 31)
        self.add_hint("Q / ESC", "Exit", x, y + 31)
        guidance = "Hold to move. Release to coast. Arrow keys also work."
        if not self.is_car:
            guidance = f"{BALANCE_TIME_SCALE:g}x speed. " + guidance
        self.text(self.base, guidance, 24, y + 62, 16, self.muted)

    def text(self, surface, label, x, y, size=18, color=None):
        surface.blit(self.fonts[size].render(label, True, color or self.ink), (x, y))

    def add_hint(self, key, label, x, y, control=None):
        key_w = self.fonts[16].size(key)[0] + 14
        rect = pygame.Rect(x, y, key_w, 22)
        pygame.draw.rect(self.base, self.key_bg, rect, border_radius=3)
        self.text(self.base, key, x + 7, y + 4, 16)
        self.text(self.base, label, rect.right + 7, y + 3, 18, self.muted)
        if control:
            self.hints.append((control, key, rect))
        return rect.right + 7 + self.fonts[18].size(label)[0] + 22

    def draw(self, env, action, controls):
        self.frame.blit(self.base, (0, 0))
        self.frame.blit(env.render(as_surface=True), (self.pad, self.header))
        values = (
            [
                f"{env.speed:+.1f} m/s",
                f"{np.degrees(env.heading):+.0f} deg",
                f"{action[1]:+.0f}",
            ]
            if self.is_car
            else [
                f"{np.degrees(env.theta):+.1f} deg",
                f"{env.x:+.2f} m",
                f"{action[0] * env.force_mag:+.0f} N",
            ]
        )
        for i, value in enumerate(values):
            self.text(
                self.frame,
                value,
                24 + i * self.cell_w,
                109,
                28,
                self.accent if i == 0 else self.ink,
            )
        active = {
            "steer": controls.action(self.is_car)[0] != 0,
            "throttle": self.is_car and action[-1] != 0,
            "straight": pygame.K_e in controls.held,
            "neutral": pygame.K_SPACE in controls.held,
        }
        for control, key, rect in self.hints:
            if active[control]:
                pygame.draw.rect(self.frame, self.accent, rect, border_radius=3)
                self.text(self.frame, key, rect.x + 7, rect.y + 4, 16, (15, 19, 24))
        if not controls.focused:
            shade = pygame.Surface(
                (self.width - self.pad * 2, self.height), pygame.SRCALPHA
            )
            shade.fill((10, 15, 22, 170))
            self.frame.blit(shade, (self.pad, self.header))
            self.text(
                self.frame,
                "PAUSED - CLICK TO RESUME",
                130,
                self.header + self.height // 2,
                28,
            )
        return self.frame


def run_demo(env, name):
    original = {key: getattr(env, key) for key in ("dt", "max_steps", "obj_type")}
    if hasattr(env, "friction"):
        original["friction"] = env.friction
    controls = KeyboardControls()
    is_car = env.action_dim == 2
    physics = FixedStepClock(time_scale=1.0 if is_car else BALANCE_TIME_SCALE)
    try:
        env.dt = physics.dt
        env.max_steps = round(original["max_steps"] * original["dt"] / env.dt)
        env.obj_type = "state"
        if "friction" in original:
            env.friction = original["friction"] ** (env.dt / original["dt"])
        env.reset(seed=0)
        pygame.display.init()
        dashboard = DemoDisplay(env)
        screen = pygame.display.set_mode(dashboard.base.get_size())
        pygame.display.set_caption(f"World Model Lab - {name}")
        pygame.key.set_repeat()
        pygame.key.stop_text_input()
        clock = pygame.time.Clock()
        previous = time.perf_counter()
        render_pending = 1.0 / RENDER_FPS
        action = np.zeros(env.action_dim, dtype=np.float32)
        if is_car:
            print(f"{name}: hold WASD / arrows to move; release to coast.")
        else:
            force = env.force_mag * BALANCE_KEY_FORCE_SCALE
            print(f"{name}: A/D or arrows apply {force:g} N; release to coast.")
            print(f"Simulation speed: {BALANCE_TIME_SCALE:g}x")
        print("R: reset | Tab: switch | Q / Esc: quit")

        while True:
            now = time.perf_counter()
            elapsed = now - previous
            previous = now
            command = controls.process(pygame.event.get())
            if command in ("quit", "switch"):
                return command == "switch"
            if command == "reset":
                env.reset()
                controls.clear()
                physics.pending = 0.0
                elapsed = 0.0
                action.fill(0)
            requested_action = controls.action(is_car)
            if controls.focused:
                for _ in range(physics.advance(elapsed)):
                    action = requested_action.copy()
                    _, _, terminated, truncated, _ = env.step(action)
                    if terminated or truncated:
                        env.reset()
                        controls.clear()
                        action.fill(0)
                        physics.pending = 0.0
                        break
            else:
                physics.pending = 0.0
                action.fill(0)
            render_pending += min(elapsed, 0.1)
            if render_pending >= 1.0 / RENDER_FPS:
                render_pending %= 1.0 / RENDER_FPS
                screen.blit(dashboard.draw(env, action, controls), (0, 0))
                pygame.display.flip()
            clock.tick(PHYSICS_HZ)
    finally:
        for key, value in original.items():
            setattr(env, key, value)
        env.close()
        pygame.display.quit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=("car", "balance"), default="car")
    args = parser.parse_args()
    name = args.env
    try:
        while True:
            env_class = CarDrivingEnv if name == "car" else BalancingEnv
            env = env_class(obs_type="state", image_size=(640, 640), max_steps=3000)
            if not run_demo(env, name):
                break
            name = "balance" if name == "car" else "car"
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
