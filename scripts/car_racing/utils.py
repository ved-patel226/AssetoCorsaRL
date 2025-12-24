import time
import collections
import numpy as np


def to_chw(frame: np.ndarray) -> np.ndarray:
    """Convert observation from HWC (RGB or grayscale) to CHW.

    Keeps a single grayscale channel if input is RGB by applying standard
    luminance weights.
    """
    f = frame[0] if isinstance(frame, tuple) else frame
    if f.ndim == 3 and f.shape[2] == 3:
        gray = np.dot(f[..., :3], [0.2989, 0.5870, 0.1140])
        gray = gray.astype(f.dtype)
        gray = np.expand_dims(gray, axis=2)
    elif f.ndim == 2:
        gray = np.expand_dims(f, axis=2)
    else:
        gray = f
    return np.transpose(gray, (2, 0, 1))


def reset_env(env):
    s = env.reset()
    return s[0] if isinstance(s, tuple) else s


def init_frame_stacks(envs, frame_stack: int):
    """Initialize per-env frame deques and return stacked states and timers."""
    stacks = [collections.deque(maxlen=frame_stack) for _ in range(len(envs))]
    start_times = [0.0 for _ in range(len(envs))]
    for i, e in enumerate(envs):
        s = reset_env(e)
        s_chw = to_chw(s)
        for _ in range(frame_stack):
            stacks[i].append(s_chw)
        start_times[i] = time.time()

    states = [np.concatenate(list(stacks[i]), axis=0) for i in range(len(envs))]
    return stacks, states, start_times
