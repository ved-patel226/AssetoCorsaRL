# Assetto Corsa — IQN RL (SPA / FormulaAlpha2022)

Short README for training and evaluating an Implicit Quantile Networks (IQN) agent in Assetto Corsa on the Spa-Francorchamps (SPA) map using FormulaAlpha2022 cars.

## Summary
Train an IQN-based RL agent to drive FormulaAlpha2022 cars on SPA using a Python environment that bridges Assetto Corsa to RL (e.g., acclient/SimBridge). Outputs: checkpoints, logs, evaluation rollouts.

## Requirements
- OS: Windows (Assetto Corsa)
- Assetto Corsa installed with SPA map and FormulaAlpha2022 car pack
- Python 3.8+
- PyTorch (1.8+)
- gym, numpy, scipy, tensorboard, stable-baselines3 (optional)
- Bridge package to control AC from Python (example: acclient or custom UDP/TCP bridge)

## Repo layout
- configs/                — YAML configs (training / eval)
- env/                    — Assetto Corsa gym wrapper
- agents/                 — IQN implementation
- scripts/                — train.py, eval.py, record.py
- checkpoints/            — saved models
- logs/                   — tensorboard logs
- README.md

## Installation
1. Clone repo into a workspace accessible to Assetto Corsa.
2. Create venv and install:
    ```
    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    ```
3. Ensure Assetto Corsa is running and the bridge (acclient) is configured to accept commands.

## Assetto Corsa Setup
- Install SPA map files into Assetto Corsa/content/tracks/.
- Install FormulaAlpha2022 car files into Assetto Corsa/content/cars/.
- Configure a dedicated session (single player, practice) or use the bridge to start sessions via Python.

## Example config (configs/iqn_spa_falpha2022.yaml)
```
env:
  track: "spa"
  car: "FormulaAlpha2022"
  weather: "clear"
  lap_start: "pit_out"

agent:
  type: "IQN"
  gamma: 0.99
  lr: 3e-4
  batch_size: 64
  n_quantiles: 32
  hidden_units: 512

train:
  total_timesteps: 5_000_000
  eval_interval: 50_000
  save_interval: 100_000
```

## Training
Start training:
```
python scripts/train.py --config configs/iqn_spa_falpha2022.yaml --out checkpoints/
```
Monitors: tensorboard logs in logs/.

Tips:
- Start with lower FPS or frame-skip to speed training.
- Use action clipping and steering normalization suited for formula cars.
- Curriculum: start with no opponents and tame weather.

## Evaluation
Run deterministic evaluation and save lap telemetry:
```
python scripts/eval.py --checkpoint checkpoints/last.pt --episodes 10 --save-rollouts runs/
```
Metrics: lap time, off-track events, collisions, average speed, cumulative reward.

## Model / Checkpoint format
- Checkpoints saved as PyTorch .pt files containing agent state_dict, optimizer state, training step, and config metadata.

## Telemetry & Replay
- scripts/record.py can record car telemetry (speed, throttle, steering, position) and optional video capture.
- Save CSV/JSON with per-step data for analysis.

## Tips for SPA & Formula Cars
- Reward shaping: combine progress on lap, lane/track position penalty, control smoothness penalty.
- Observation: include speed, heading error to apex, distance to track borders, gear, angular velocity.
- Use frame stacking or LSTM if using visual inputs.

## Troubleshooting
- Bridge connection errors: verify Assetto Corsa running and bridge port open.
- Physics mismatch: tune action scaling and controller frequency.
- Training instabilities: reduce lr, increase batch_size, or normalize observations.

## License & Contact
- License: MIT (adjust per project)
- Maintainer: project team (update with email or repo link)

Produced by GitHub Copilot.