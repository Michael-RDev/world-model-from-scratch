"""Regression checks for released keys and the interactive frame loop."""

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pygame
from envs import demo


def key(kind, code, **kwargs):
    return pygame.event.Event(kind, key=code, **kwargs)


class ControlsTests(unittest.TestCase):
    def test_releasing_throttle_keeps_independently_held_steering(self):
        controls = demo.KeyboardControls()
        controls.process([key(pygame.KEYDOWN, pygame.K_w), key(pygame.KEYDOWN, pygame.K_a)])
        np.testing.assert_array_equal(controls.action(True), [1, 1])
        controls.process([key(pygame.KEYUP, pygame.K_w)])
        np.testing.assert_array_equal(controls.action(True), [1, 0])
        controls.process([key(pygame.KEYUP, pygame.K_a)])
        np.testing.assert_array_equal(controls.action(True), [0, 0])

    def test_cart_force_stops_on_release(self):
        controls = demo.KeyboardControls()
        for code, force in ((pygame.K_a, -1.0), (pygame.K_d, 1.0)):
            controls.process([key(pygame.KEYDOWN, code)])
            np.testing.assert_array_equal(controls.action(False), [force])
            controls.process([key(pygame.KEYUP, code)])
            np.testing.assert_array_equal(controls.action(False), [0])

    def test_opposed_keys_cancel_and_arrow_aliases_work(self):
        controls = demo.KeyboardControls()
        controls.process([key(pygame.KEYDOWN, k) for k in (pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT)])
        np.testing.assert_array_equal(controls.action(True), [1, 0])
        controls.process([key(pygame.KEYUP, pygame.K_DOWN)])
        np.testing.assert_array_equal(controls.action(True), [1, 1])

    def test_focus_loss_clears_keys_even_when_keyup_is_missed(self):
        controls = demo.KeyboardControls()
        controls.process([key(pygame.KEYDOWN, pygame.K_w)])
        controls.process([pygame.event.Event(pygame.WINDOWFOCUSLOST)])
        np.testing.assert_array_equal(controls.action(True), [0, 0])
        controls.process([pygame.event.Event(pygame.WINDOWFOCUSGAINED)])
        np.testing.assert_array_equal(controls.action(True), [0, 0])

    def test_repeat_does_not_repeatedly_reset_or_switch(self):
        controls = demo.KeyboardControls()
        self.assertEqual(controls.process([key(pygame.KEYDOWN, pygame.K_r)]), "reset")
        self.assertIsNone(controls.process([key(pygame.KEYDOWN, pygame.K_r, repeat=True)]))
        self.assertEqual(controls.process([key(pygame.KEYDOWN, pygame.K_TAB)]), "switch")
        self.assertEqual(controls.process([pygame.event.Event(pygame.QUIT)]), "quit")

    def test_cart_can_arrest_a_falling_pole_in_both_directions(self):
        for side, code in ((-1, pygame.K_a), (1, pygame.K_d)):
            env = demo.BalancingEnv(obs_type="state", seed=0)
            env.dt = 1 / demo.PHYSICS_HZ
            env.x = env.x_dot = 0.0
            env.theta = side * np.deg2rad(25)
            env.theta_dot = side * 1.0
            controls = demo.KeyboardControls()
            controls.process([key(pygame.KEYDOWN, code)])
            # A fixed 0.2-second push toward the lean must reverse the fall.
            # The previous half-force keyboard mapping cannot do this.
            for _ in range(24):
                _, _, terminated, truncated, _ = env.step(controls.action(False))
                self.assertFalse(terminated or truncated)
            self.assertLess(side * env.theta_dot, 0)
            self.assertGreater(side * env.x, 0)
            self.assertLess(abs(env.theta), np.deg2rad(35))


class TimingTests(unittest.TestCase):
    def test_simulated_time_is_independent_of_render_rate(self):
        for scale, expected in ((1.0, 360), (0.5, 180)):
            for fps in (30, 60, 144):
                clock = demo.FixedStepClock(time_scale=scale)
                self.assertEqual(sum(clock.advance(1 / fps) for _ in range(fps * 3)), expected)

    def test_stall_cannot_create_unbounded_catchup(self):
        for scale, expected in ((1.0, 12), (0.5, 6)):
            clock = demo.FixedStepClock(time_scale=scale)
            self.assertEqual(clock.advance(5.0), expected)
            self.assertEqual(clock.advance(0), 0)


class LoopTests(unittest.TestCase):
    def exercise_loop(self, cls, press, expected):
        env = cls(obs_type="image", image_size=(64, 64), max_steps=100)
        original = (env.dt, env.max_steps, env.obj_type)
        old_friction = getattr(env, "friction", None)
        events = [
            [key(pygame.KEYDOWN, press)],
            [],
            [key(pygame.KEYUP, press)],
            [],
            [key(pygame.KEYDOWN, pygame.K_q)],
        ]
        steps = []
        step = env.step

        def record(action):
            self.assertEqual(env.obj_type, "state")
            self.assertAlmostEqual(env.dt, 1 / 120)
            if old_friction is not None:
                self.assertAlmostEqual(env.friction ** 120, old_friction ** 10)
            steps.append(action.copy())
            return step(action)

        fake_clock = type("Clock", (), {"tick": lambda self, fps: 0})()
        with patch.object(demo.pygame.event, "get", side_effect=events), \
             patch.object(demo.time, "perf_counter", side_effect=[i / 60 for i in range(6)]), \
             patch.object(demo.pygame.time, "Clock", return_value=fake_clock), \
             patch.object(env, "step", side_effect=record):
            self.assertFalse(demo.run_demo(env, "test"))
        # Two rendered frames of held input, then two of released input.
        held_steps = 4 if env.action_dim == 2 else 2
        self.assertEqual(len(steps), held_steps * 2)
        np.testing.assert_array_equal(steps[:held_steps], np.tile(expected, (held_steps, 1)))
        np.testing.assert_array_equal(steps[held_steps:], np.zeros((held_steps, env.action_dim)))
        self.assertEqual((env.dt, env.max_steps, env.obj_type), original)
        if old_friction is not None:
            self.assertEqual(env.friction, old_friction)
        self.assertFalse(pygame.display.get_init())

    def test_car_loop_processes_release_and_restores_training_settings(self):
        self.exercise_loop(demo.CarDrivingEnv, pygame.K_w, [0, 1])

    def test_balance_loop_processes_release_and_restores_training_settings(self):
        self.exercise_loop(demo.BalancingEnv, pygame.K_a, [-1.0])

    def test_pole_moves_freely_without_input_or_hidden_corrections(self):
        env = demo.BalancingEnv(obs_type="state", image_size=(64, 64), max_steps=100)
        events = [
            [],
            [key(pygame.KEYDOWN, pygame.K_h)],
            [],
            [],
            [key(pygame.KEYDOWN, pygame.K_q)],
        ]
        actions = []
        angles = []
        original_step = env.step

        def record(action):
            actions.append(action.copy())
            angles.append(env.theta)
            return original_step(action)

        fake_clock = type("Clock", (), {"tick": lambda self, fps: 0})()
        with patch.object(demo.pygame.event, "get", side_effect=events), \
             patch.object(demo.time, "perf_counter", side_effect=[i / 60 for i in range(6)]), \
             patch.object(demo.pygame.time, "Clock", return_value=fake_clock), \
             patch.object(env, "step", side_effect=record):
            self.assertFalse(demo.run_demo(env, "test"))
        self.assertEqual(len(actions), 4)
        np.testing.assert_array_equal(actions, np.zeros((4, 1)))
        self.assertGreater(abs(angles[-1]), abs(angles[0]))

    def test_surface_path_matches_rgb_observations(self):
        for cls in (demo.CarDrivingEnv, demo.BalancingEnv):
            env = cls(image_size=(64, 64), seed=0)
            rgb = env.render()
            surface = env.render(as_surface=True)
            np.testing.assert_array_equal(rgb, pygame.surfarray.array3d(surface).transpose(1, 0, 2))


if __name__ == "__main__":
    unittest.main()
