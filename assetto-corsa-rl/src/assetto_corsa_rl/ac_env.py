import gymnasium as gym
import time

import pyvjoy

j = pyvjoy.VJoyDevice(1)
time.sleep(1)
j.set_button(15, 1)
j.update()
time.sleep(1)

j.set_button(15, 0)
j.update()
