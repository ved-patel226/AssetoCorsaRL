"""Test Xbox controller emulation via ViGEm for Assetto Corsa.

This uses pyvjoystick's vigem module to create a virtual Xbox 360 controller,
which is better supported by Assetto Corsa than VJoy.

Install: pip install pyvjoystick
"""

import time
import sys


def test_xbox_connection():
    """Test Xbox controller connection and basic functionality."""

    print("=" * 60)
    print("Xbox Controller (ViGEm) Connection Test")
    print("=" * 60)

    # Step 1: Check if pyvjoystick is installed
    print("\n[1/4] Checking pyvjoystick installation...")
    try:
        from pyvjoystick import vigem as vg

        print("✓ pyvjoystick (vigem) is installed")
    except ImportError as e:
        print(f"❌ pyvjoystick is NOT installed: {e}")
        print("\nTo install pyvjoystick:")
        print("  pip install pyvjoystick")
        print("\nAlso make sure ViGEmBus driver is installed:")
        print("  Download from: https://github.com/ViGEm/ViGEmBus/releases")
        return None

    # Step 2: Try to create gamepad
    print("\n[2/4] Creating virtual Xbox 360 gamepad...")
    try:
        gamepad = vg.VX360Gamepad()
        print("✓ Virtual Xbox 360 gamepad connected")
        print("   (Check Windows Settings > Devices > Bluetooth & other devices)")
    except Exception as e:
        print(f"❌ Failed to create gamepad: {e}")
        print("\nPossible issues:")
        print("  1. ViGEmBus driver not installed")
        print("  2. Driver not running properly")
        print("  3. Insufficient permissions (try running as administrator)")
        return None

    # Step 3: Test button
    print("\n[3/4] Testing A button...")
    try:
        gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        gamepad.update()
        print("✓ A button pressed")
        time.sleep(1)

        gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        gamepad.update()
        print("✓ A button released")
    except Exception as e:
        print(f"❌ Failed to press button: {e}")
        return None

    # Step 4: Test triggers and joysticks
    print("\n[4/4] Testing triggers and joysticks...")
    try:
        # Test left trigger (brake)
        gamepad.left_trigger(value=128)
        gamepad.update()
        time.sleep(0.3)

        # Test right trigger (throttle)
        gamepad.right_trigger(value=255)
        gamepad.update()
        time.sleep(0.3)

        # Test left joystick (steering)
        gamepad.left_joystick(x_value=-16000, y_value=0)
        gamepad.update()
        time.sleep(0.3)

        # Reset everything
        gamepad.reset()
        gamepad.update()
        print("✓ Triggers and joysticks working")
    except Exception as e:
        print(f"❌ Failed to control axes: {e}")
        return None

    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nYour Xbox controller emulation is working correctly.")
    return gamepad


def test_racing_controls(gamepad, duration=5):
    """Test racing-specific controls (steering, throttle, brake)."""

    print("\n" + "=" * 60)
    print("Racing Controls Test")
    print("=" * 60)

    print("\nTesting racing controls (for Assetto Corsa)...")
    print("  Left Joystick X = Steering")
    print("  Right Trigger = Throttle")
    print("  Left Trigger = Brake")

    # Test 1: Steering
    print("\n[Steering Test]")
    positions = [
        (0, "Center"),
        (-32768, "Full Left"),
        (0, "Center"),
        (32767, "Full Right"),
        (0, "Center"),
    ]

    for x_value, label in positions:
        print(f"  {label}...")
        gamepad.left_joystick(x_value=x_value, y_value=0)
        gamepad.update()
        time.sleep(0.5)

    # Test 2: Throttle
    print("\n[Throttle Test]")
    print("  Ramping up...")
    for value in range(0, 256, 32):
        gamepad.right_trigger(value=value)
        gamepad.update()
        time.sleep(0.1)

    print("  Ramping down...")
    for value in range(255, -1, -32):
        gamepad.right_trigger(value=value)
        gamepad.update()
        time.sleep(0.1)

    # Test 3: Brake
    print("\n[Brake Test]")
    print("  Applying brake...")
    gamepad.left_trigger(value=255)
    gamepad.update()
    time.sleep(0.5)

    print("  Releasing brake...")
    gamepad.left_trigger(value=0)
    gamepad.update()

    # Test 4: Combined (steering + throttle)
    print("\n[Combined Test]")
    print("  Steering while accelerating...")
    import math

    for i in range(30):
        # Sine wave steering
        steer = int(math.sin(i * 0.3) * 20000)
        # Constant throttle
        throttle = 180

        gamepad.left_joystick(x_value=steer, y_value=0)
        gamepad.right_trigger(value=throttle)
        gamepad.update()
        time.sleep(0.1)

    # Reset
    gamepad.reset()
    gamepad.update()
    print("\n✓ Racing controls test completed!")


def test_racing_controls_float(gamepad):
    """Test racing controls using float API (easier to use)."""

    print("\n" + "=" * 60)
    print("Racing Controls Test (Float API)")
    print("=" * 60)

    print("\nUsing float values (-1.0 to 1.0 for steering, 0.0 to 1.0 for triggers)...")

    import math

    # Test smooth steering and throttle
    print("\n[Smooth Driving Simulation]")
    for i in range(60):
        t = i / 60.0

        # Steering: gentle sine wave (-0.5 to 0.5)
        steer = math.sin(t * 4 * math.pi) * 0.5

        # Throttle: varying (0.4 to 0.8)
        throttle = 0.6 + 0.2 * math.sin(t * 2 * math.pi)

        gamepad.left_joystick_float(x_value_float=steer, y_value_float=0.0)
        gamepad.right_trigger_float(value_float=throttle)
        gamepad.update()

        if i % 10 == 0:
            print(f"  Step {i}: Steer={steer:+.2f}, Throttle={throttle:.2f}")

        time.sleep(0.1)

    # Reset
    gamepad.reset()
    gamepad.update()
    print("\n✓ Float API test completed!")


def test_assetto_corsa_driving():
    """Full Assetto Corsa driving test."""

    print("\n" + "=" * 60)
    print("Assetto Corsa Driving Test")
    print("=" * 60)

    print("\n⚠️  SETUP:")
    print("   1. Launch Assetto Corsa")
    print("   2. Go to Options > Controls")
    print("   3. Select 'Xbox 360 Controller' (should auto-detect)")
    print("   4. Start a practice session")
    print("   5. Be on track and ready to drive")
    print()

    response = input("Ready to start? [Y/n]: ").strip().lower()
    if response and response not in ["y", "yes"]:
        print("Test cancelled.")
        return

    # Create gamepad
    print("\n🎮 Creating Xbox controller...")
    try:
        from pyvjoystick import vigem as vg

        gamepad = vg.VX360Gamepad()
        print("✓ Xbox 360 controller connected")
    except Exception as e:
        print(f"❌ Failed: {e}")
        return

    print("\nStarting driving test in 3 seconds...")
    print("Watch Assetto Corsa - the car should move!")

    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)

    print("\n🏁 GO!\n")

    import math

    try:
        # Test 1: Throttle only
        print("▶ Test 1: Acceleration (5s)")
        for i in range(50):
            throttle = min(1.0, i / 25.0)  # Ramp to 100% then hold
            gamepad.right_trigger_float(value_float=throttle)
            gamepad.update()
            time.sleep(0.1)

        gamepad.reset()
        gamepad.update()
        time.sleep(1)

        # Test 2: Steering sweep
        print("\n▶ Test 2: Steering Sweep (6s)")
        for i in range(60):
            steer = math.sin(i * 0.2) * 0.8  # -0.8 to +0.8
            throttle = 0.5

            gamepad.left_joystick_float(x_value_float=steer, y_value_float=0.0)
            gamepad.right_trigger_float(value_float=throttle)
            gamepad.update()
            time.sleep(0.1)

        gamepad.reset()
        gamepad.update()
        time.sleep(1)

        # Test 3: Combined driving
        print("\n▶ Test 3: Combined Driving (10s)")
        for i in range(100):
            t = i / 100.0

            # Realistic steering
            steer = math.sin(t * 3 * math.pi) * 0.3

            # Varying throttle (50-80%)
            throttle = 0.65 + 0.15 * math.sin(t * 5 * math.pi)

            gamepad.left_joystick_float(x_value_float=steer, y_value_float=0.0)
            gamepad.right_trigger_float(value_float=throttle)
            gamepad.update()

            if i % 20 == 0:
                print(f"  Step {i}: Steer={steer:+.2f}, Throttle={throttle:.2f}")

            time.sleep(0.1)

        # Test 4: Braking
        print("\n▶ Test 4: Braking (3s)")
        # Accelerate first
        gamepad.right_trigger_float(value_float=0.8)
        gamepad.update()
        time.sleep(1.5)

        # Brake
        gamepad.right_trigger_float(value_float=0.0)
        gamepad.left_trigger_float(value_float=1.0)
        gamepad.update()
        time.sleep(1.5)

        # Reset
        gamepad.reset()
        gamepad.update()

        print("\n" + "=" * 60)
        print("✓ ALL DRIVING TESTS COMPLETE")
        print("=" * 60)

        print("\nDid the car move in Assetto Corsa?")
        print("  YES → Xbox controller emulation is working! ✓")
        print("  NO  → Check AC detected the Xbox controller in Controls settings")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted")
    finally:
        print("\n🔄 Resetting controller...")
        gamepad.reset()
        gamepad.update()
        print("✓ Controller reset")
        # Note: gamepad disconnects when object is destroyed


def main():
    """Main test menu."""

    print("Xbox Controller (ViGEm) Test Menu")
    print("=" * 60)
    print("Choose a test:")
    print("  1. Basic connection test (recommended first)")
    print("  2. Racing controls test (steering, throttle, brake)")
    print("  3. Racing controls test (float API)")
    print("  4. Full Assetto Corsa driving test")
    print("  5. All tests")
    print()

    choice = input("Enter choice [1-5] or press Enter for option 1: ").strip()

    if not choice:
        choice = "1"

    if choice == "1":
        gamepad = test_xbox_connection()
    elif choice == "2":
        gamepad = test_xbox_connection()
        if gamepad:
            test_racing_controls(gamepad)
    elif choice == "3":
        gamepad = test_xbox_connection()
        if gamepad:
            test_racing_controls_float(gamepad)
    elif choice == "4":
        test_assetto_corsa_driving()
    elif choice == "5":
        gamepad = test_xbox_connection()
        if gamepad:
            test_racing_controls(gamepad)
            test_racing_controls_float(gamepad)
        print("\n⏸️  Waiting 3 seconds before AC test...")
        time.sleep(3)
        test_assetto_corsa_driving()
    else:
        print("Invalid choice. Running default test...")
        test_xbox_connection()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Test interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
