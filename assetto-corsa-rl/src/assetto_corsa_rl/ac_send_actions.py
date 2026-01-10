"""Controller interface to send Xbox controller input to Assetto Corsa.

Provides a safe wrapper around pyvjoystick (ViGEm) with both integer and
float APIs, a dummy fallback for non-Windows or missing-driver environments,
context-manager support, and convenience helpers used by tests.
"""

from __future__ import annotations

from typing import Optional


class _DummyGamepad:
    """A no-op gamepad used when ViGEm is unavailable (e.g., on CI).

    Methods match the pyvjoystick API used in tests so the rest of the code
    can call into this object without conditionals.
    """

    def press_button(self, *_, **__):
        return

    def release_button(self, *_, **__):
        return

    def left_trigger(self, value: int = 0):
        return

    def right_trigger(self, value: int = 0):
        return

    def left_joystick(self, x_value: int = 0, y_value: int = 0):
        return

    def left_joystick_float(
        self, x_value_float: float = 0.0, y_value_float: float = 0.0
    ):
        return

    def right_trigger_float(self, value_float: float = 0.0):
        return

    def update(self):
        return

    def reset(self):
        return


class XboxController:
    """High-level wrapper for an Xbox 360 virtual gamepad (ViGEm via pyvjoystick).

    Example:
        with XboxController() as gp:
            gp.press_a()
            gp.update()

    If ViGEm is not present or pyvjoystick is not installed, a dummy gamepad
    is used so code remains testable on non-Windows platforms.
    """

    def __init__(self, use_dummy: bool = False):
        self._use_dummy = use_dummy
        self._vg = None
        self._gamepad = None
        self._init_gamepad()

    def _init_gamepad(self):
        if self._use_dummy:
            self._gamepad = _DummyGamepad()
            return

        try:
            from pyvjoystick import vigem as vg  # type: ignore

            self._vg = vg
            self._gamepad = vg.VX360Gamepad()
        except Exception:
            # Fallback to a dummy object when ViGEm or driver is unavailable.
            self._gamepad = _DummyGamepad()

    # Convenience methods matching test_xbox_controller usage
    def press_button(self, button):
        self._gamepad.press_button(button=button)

    def release_button(self, button):
        self._gamepad.release_button(button=button)

    def press_a(self):
        if self._vg:
            self._gamepad.press_button(button=self._vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        else:
            # no-op on dummy
            self._gamepad.press_button(None)

    def release_a(self):
        if self._vg:
            self._gamepad.release_button(button=self._vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        else:
            self._gamepad.release_button(None)

    def left_trigger(self, value: int):
        """Set left trigger using integer value 0-255."""
        self._gamepad.left_trigger(value=value)

    def right_trigger(self, value: int):
        """Set right trigger using integer value 0-255."""
        self._gamepad.right_trigger(value=value)

    def left_joystick(self, x_value: int, y_value: int):
        """Set left joystick using integer values (e.g., -32768..32767)."""
        self._gamepad.left_joystick(x_value=x_value, y_value=y_value)

    # Float APIs convenience (normalized ranges)
    def left_joystick_float(self, x_value_float: float, y_value_float: float):
        """Set left joystick using floats in range [-1.0, 1.0]."""
        # Convert to int range used by pyvjoystick (approx)
        max_int = 32767
        xi = int(max(-1.0, min(1.0, x_value_float)) * max_int)
        yi = int(max(-1.0, min(1.0, y_value_float)) * max_int)
        if hasattr(self._gamepad, "left_joystick_float"):
            # prefer native float API if available
            try:
                self._gamepad.left_joystick_float(
                    x_value_float=x_value_float, y_value_float=y_value_float
                )
                return
            except Exception:
                pass
        self._gamepad.left_joystick(x_value=xi, y_value=yi)

    def right_trigger_float(self, value_float: float):
        """Set right trigger using float in range [0.0, 1.0]."""
        vi = int(max(0.0, min(1.0, value_float)) * 255.0)
        if hasattr(self._gamepad, "right_trigger_float"):
            try:
                self._gamepad.right_trigger_float(value_float=value_float)
                return
            except Exception:
                pass
        self._gamepad.right_trigger(value=vi)

    def left_trigger_float(self, value_float: float):
        """Set left trigger using float in range [0.0, 1.0]."""
        vi = int(max(0.0, min(1.0, value_float)) * 255.0)
        self._gamepad.left_trigger(value=vi)

    def update(self):
        self._gamepad.update()

    def reset(self):
        self._gamepad.reset()

    # Context manager support
    def __enter__(self) -> "XboxController":
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.reset()
            self.update()
        except Exception:
            pass


# Small convenience factory used by upper-level code
def create_controller(use_dummy: bool = False) -> XboxController:
    """Create and return a configured XboxController instance."""
    return XboxController(use_dummy=use_dummy)
