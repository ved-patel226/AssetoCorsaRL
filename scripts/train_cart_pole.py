import os
import sys

# Ensure project root is on sys.path so `models` can be imported
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.agent.iqn_agent import IQN_Agent


# Example usage:
# agent = IQN_Agent(...)
