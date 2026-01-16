"""Load a trained VAE and inspect encoder/decoder outputs on environment frames.

Example:
    python assetto-corsa-rl/scripts/ac/load_vae.py --ckpt checkpoints/vae-final.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import torch
import torchvision
import numpy as np
from PIL import Image
import cv2
import time

repo_root = Path(__file__).resolve().parents[2]
src_path = str(repo_root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from assetto_corsa_rl.model.vae import ConvVAE  # type: ignore
from assetto_corsa_rl.env_helper import create_gym_env  # type: ignore
from collections import deque

# optional AC env
try:
    from assetto_corsa_rl.ac_env import make_env as make_ac_env  # type: ignore
except Exception:
    make_ac_env = None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt", type=Path, required=True, help="VAE checkpoint (state_dict) to load"
    )
    p.add_argument(
        "--env",
        type=str,
        choices=["gym", "ac"],
        default="ac",
        help="Which environment to use: 'gym' (CarRacing) or 'ac' (Assetto Corsa)",
    )
    p.add_argument(
        "--frames",
        type=int,
        default=1,
        help="Number of grayscale frames stacked by the env",
    )
    p.add_argument(
        "--image-shape", type=str, default="84x84", help="HxW image shape (e.g. 84x84)"
    )
    p.add_argument(
        "--device", type=str, default=None, help="Device to run on (cpu/cuda)"
    )
    return p.parse_args()


def _parse_shape(s: str) -> Tuple[int, int]:
    h, w = s.lower().split("x")
    return int(h), int(w)


def graystack_to_rgb_input(pixels: torch.Tensor, frames: int):
    """Convert grayscale stacked frames (C=frames) to VAE input with 3*frames channels.

    Accepts pixels shaped (C,H,W) or (B,C,H,W) and returns (B, 3*frames, H, W).
    """
    if pixels.dim() == 3:
        pixels = pixels.unsqueeze(0)
    B, C, H, W = pixels.shape
    assert C == frames, f"Expected {frames} frames in pixels, got {C}"
    # Expand each frame to 3 channels and concatenate
    per_frame = []
    for i in range(frames):
        fr = pixels[:, i : i + 1, :, :].repeat(1, 3, 1, 1)  # (B,3,H,W)
        per_frame.append(fr)
    x = torch.cat(per_frame, dim=1)
    return x


def _recon_display(recon_rgb: torch.Tensor, title: str = "VAE") -> int:
    """Display only the decoder output (reconstruction) in a popout window. Returns pressed key code."""
    # recon_rgb expected (C,H,W) or (1,C,H,W)
    if recon_rgb.dim() == 4:
        img_t = recon_rgb[0]
    else:
        img_t = recon_rgb
    # ensure first 3 channels if more
    if img_t.size(0) > 3:
        img_t = img_t[-3:]
    # move to cpu and to numpy H,W,C scaled 0..255
    arr = (img_t.detach().cpu().numpy() * 255.0).astype(np.uint8)
    arr = np.transpose(arr, (1, 2, 0))  # H,W,C
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    # convert RGB->BGR for cv2
    arr = arr[..., ::-1]

    cv2.imshow(title, arr)
    key = cv2.waitKey(1) & 0xFF
    return key


def main():
    args = parse_args()
    img_h, img_w = _parse_shape(args.image_shape)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # instantiate model matching training defaults
    in_channels = 3 * args.frames
    vae = ConvVAE(in_channels=in_channels, im_shape=(img_h, img_w))
    ckpt = torch.load(str(args.ckpt), map_location="cpu")
    try:
        vae.load_state_dict(ckpt)
    except Exception:
        # try nested key (lightning or dict)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            vae.load_state_dict(ckpt["state_dict"])  # type: ignore
        else:
            raise

    vae.to(device)
    vae.eval()

    # continuous interactive-only display of decoder output
    if args.env == "gym":
        env = create_gym_env(height=img_h, width=img_w, device=device, num_envs=1)
        td = env.reset()

        try:
            cv2.namedWindow("VAE", cv2.WINDOW_NORMAL)
            while True:
                pixels = td["pixels"]  # shape (B, C, H, W) or (C, H, W)
                pixels_sample = pixels[0] if pixels.dim() == 4 else pixels

                x = graystack_to_rgb_input(pixels_sample.cpu(), args.frames).to(device)
                with torch.no_grad():
                    recon, _, _ = vae(x)

                # show only decoder output
                recon_img = recon[0, :3, :, :] if recon.size(1) >= 3 else recon[0]
                key = _recon_display(recon_img, title="VAE")
                if key in (ord("q"), 27):
                    break

                # step env
                action_shape = env.action_spec.shape
                action = torch.zeros((env.num_workers, *action_shape), device=device)
                td = env.step(action)

                time.sleep(0.01)
        except KeyboardInterrupt:
            pass
        finally:
            cv2.destroyAllWindows()

    else:
        if make_ac_env is None:
            raise RuntimeError("Assetto Corsa env not available (failed to import).")

        # create AC environment: returns gym-style obs dicts
        env = make_ac_env(include_image=True, observation_image_shape=(img_h, img_w))
        obs, info = env.reset()

        # initialize frame buffer with the initial frame repeated
        initial_img = obs["image"]  # H, W, 1 uint8
        frame_buf = deque(maxlen=args.frames)
        fr = torch.from_numpy(initial_img[..., 0].astype(np.float32) / 255.0)
        for _ in range(args.frames):
            frame_buf.append(fr.clone())

        try:
            cv2.namedWindow("VAE", cv2.WINDOW_NORMAL)
            while True:
                stacked = torch.stack(list(frame_buf), dim=0)
                x = graystack_to_rgb_input(stacked, args.frames).to(device)

                with torch.no_grad():
                    recon, _, _ = vae(x)

                recon_img = recon[0, :3, :, :] if recon.size(1) >= 3 else recon[0]
                key = _recon_display(recon_img, title="VAE")
                if key in (ord("q"), 27):
                    break

                # step env with zero action to advance (steer, throttle, brake)
                action = np.zeros(3, dtype=np.float32)
                obs, reward, terminated, truncated, info = env.step(action)

                next_img = obs["image"]  # H,W,1
                next_fr = torch.from_numpy(next_img[..., 0].astype(np.float32) / 255.0)
                frame_buf.append(next_fr)

                if terminated or truncated:
                    obs, info = env.reset()
                    initial_img = obs["image"]
                    frame_buf = deque(
                        [
                            torch.from_numpy(
                                initial_img[..., 0].astype(np.float32) / 255.0
                            )
                        ]
                        * args.frames,
                        maxlen=args.frames,
                    )

                time.sleep(0.01)
        except KeyboardInterrupt:
            pass
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
