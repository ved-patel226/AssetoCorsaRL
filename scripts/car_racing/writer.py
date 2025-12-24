import wandb


class WandbWriter:
    """Simple WandB wrapper that keeps an internal step counter.

    This isolates WandB usage so the trainer stays focused on training logic.
    """

    def __init__(self, wandb_module=wandb):
        self._step = 0
        self.wandb = wandb_module

    def _next_step(self, step):
        if step is None:
            self._step += 1
        else:
            if step <= self._step:
                self._step += 1
            else:
                self._step = step
        return self._step

    def add_scalar(self, tag, value, step=None, **kwargs):
        s = self._next_step(step)
        try:
            self.wandb.log({tag: value}, step=s)
        except Exception:
            pass

    def log_dict(self, data, step=None):
        s = self._next_step(step)
        try:
            self.wandb.log(data, step=s)
        except Exception:
            pass
