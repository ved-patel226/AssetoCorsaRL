"""Interactive tests for Xbox controller buttons using ViGEm (pyvjoystick.vigem).

Usage:
    python tests/test_xbox_buttons.py

Features:
- Sequential button press with optional user confirmation
- Press-all-buttons hold test for a specified duration
- Resets controller state and cleans up on exit

Requires:
- pip install pyvjoystick
- ViGEmBus driver installed
"""

import time
import sys


def _get_button_members(vg):
    """Return (name, member) pairs for XUSB_BUTTON enum in vg."""
    try:
        members = []
        # If enum has __members__ (Enum), iterate that
        if hasattr(vg.XUSB_BUTTON, "__members__"):
            for name, member in vg.XUSB_BUTTON.__members__.items():
                members.append((name, member))
        else:
            # Fallback: gather attributes that start with XUSB_GAMEPAD_
            for name in dir(vg.XUSB_BUTTON):
                if name.startswith("XUSB_GAMEPAD_"):
                    try:
                        members.append((name, getattr(vg.XUSB_BUTTON, name)))
                    except Exception:
                        pass
        return members
    except Exception:
        return []


def sequential_button_test(wait=0.6, prompt=True):
    """Press each Xbox button sequentially and optionally ask for visual confirmation."""
    print("\nXbox Sequential Button Test")
    print("=" * 50)

    try:
        from pyvjoystick import vigem as vg
    except ImportError:
        print("❌ pyvjoystick not installed. Run: pip install pyvjoystick")
        return False

    try:
        gamepad = vg.VX360Gamepad()
    except Exception as e:
        print(f"❌ Could not create virtual gamepad: {e}")
        return False

    members = _get_button_members(vg)
    if not members:
        print("⚠ Could not discover XUSB_BUTTON members - falling back to common list")
        members = [
            ("XUSB_GAMEPAD_A", vg.XUSB_BUTTON.XUSB_GAMEPAD_A),
            ("XUSB_GAMEPAD_B", vg.XUSB_BUTTON.XUSB_GAMEPAD_B),
            ("XUSB_GAMEPAD_X", vg.XUSB_BUTTON.XUSB_GAMEPAD_X),
            ("XUSB_GAMEPAD_Y", vg.XUSB_BUTTON.XUSB_GAMEPAD_Y),
            ("XUSB_GAMEPAD_LEFT_SHOULDER", vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER),
            ("XUSB_GAMEPAD_RIGHT_SHOULDER", vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER),
            ("XUSB_GAMEPAD_BACK", vg.XUSB_BUTTON.XUSB_GAMEPAD_BACK),
            ("XUSB_GAMEPAD_START", vg.XUSB_BUTTON.XUSB_GAMEPAD_START),
            ("XUSB_GAMEPAD_LEFT_THUMB", vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_THUMB),
            ("XUSB_GAMEPAD_RIGHT_THUMB", vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_THUMB),
            ("XUSB_GAMEPAD_DPAD_UP", vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP),
            ("XUSB_GAMEPAD_DPAD_DOWN", vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN),
            ("XUSB_GAMEPAD_DPAD_LEFT", vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT),
            ("XUSB_GAMEPAD_DPAD_RIGHT", vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT),
        ]

    try:
        print(
            "Open Game Controllers (joy.cpl) -> select the Xbox controller -> Properties"
        )
        print(
            "Watch the button indicator and answer prompts to verify each button (or run with --no-prompt)"
        )
        for name, member in members:
            print(f"\nPressing {name}...")
            try:
                gamepad.press_button(button=member)
                gamepad.update()
            except Exception as e:
                print(f"  ❌ Failed to press {name}: {e}")
                continue

            time.sleep(wait)

            try:
                gamepad.release_button(button=member)
                gamepad.update()
            except Exception as e:
                print(f"  ❌ Failed to release {name}: {e}")

            if prompt:
                resp = input(f"  Did you see {name} light up? [y/N]: ")
                if resp.lower().strip() == "y":
                    print("   ✓")
                else:
                    print("   ⚠ Not observed")

        print("\nResetting controller...")
        gamepad.reset()
        gamepad.update()
        print("✓ Sequential button test completed")
        return True

    finally:
        try:
            gamepad.reset()
            gamepad.update()
        except Exception:
            pass


def press_all_buttons(duration=3):
    """Press all discovered buttons simultaneously for duration seconds, then release."""
    print("\nXbox Press-All Buttons Test")
    print("=" * 50)

    try:
        from pyvjoystick import vigem as vg
    except ImportError:
        print("❌ pyvjoystick not installed. Run: pip install pyvjoystick")
        return False

    try:
        gamepad = vg.VX360Gamepad()
    except Exception as e:
        print(f"❌ Could not create virtual gamepad: {e}")
        return False

    members = _get_button_members(vg)
    if not members:
        print("⚠ Could not discover XUSB_BUTTON members - nothing to press")
        return False

    try:
        print(f"Pressing {len(members)} buttons for {duration}s - watch joy.cpl")
        for name, member in members:
            try:
                gamepad.press_button(button=member)
            except Exception as e:
                print(f"  ❌ Failed to press {name}: {e}")

        gamepad.update()
        for i in range(duration, 0, -1):
            print(f"  Holding... {i}s", end="\r")
            time.sleep(1)
        print("\nReleasing all buttons...")

        for name, member in members:
            try:
                gamepad.release_button(button=member)
            except Exception as e:
                print(f"  ❌ Failed to release {name}: {e}")
        gamepad.update()

        gamepad.reset()
        gamepad.update()
        print("✓ Press-all test completed")
        return True

    finally:
        try:
            gamepad.reset()
            gamepad.update()
        except Exception:
            pass


def main():
    print("Xbox Buttons Test")
    print("=" * 40)
    print("Choose mode:")
    print("  1. Sequential (interactive)  - asks you to confirm each button")
    print("  2. Press all buttons simultaneously (hold for a few seconds)")
    print("  3. Sequential without prompts (--no-prompt)")
    print()

    choice = input("Enter choice [1-3, default=1]: ").strip()
    if not choice:
        choice = "1"

    if choice == "1":
        sequential_button_test(wait=0.6, prompt=True)
    elif choice == "2":
        press_all_buttons(duration=5)
    elif choice == "3":
        sequential_button_test(wait=0.4, prompt=False)
    else:
        print("Invalid choice")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
