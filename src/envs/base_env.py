import numpy as np

"""This code is going to have a lot of typing cause ruff and neovim dont like when its not, plus its good practice"""


class ENV2D:
    def __init__(
        self,
        obj_type: str = "image",
        image_size: tuple = (64, 64),
        max_steps: int = 500,
        seed: int | None = None,
    ):
        assert obj_type in ("image", "state"), "obj_type has to be in image or state"

        self.obj_type = obj_type
        self.img_size = image_size
        self.max_steps = max_steps
        self.step_count = 0
        self.np_random = np.random.default_rng(seed)

    def seed(self, seed: int | None = None):
        self.np_random = np.random.default_rng(seed)

    def reset(self, seed: int | None = None):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def clip_action(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        return np.clip(action, self.action_low, self.action_high)


# implement this later
class Env3D:
    pass
