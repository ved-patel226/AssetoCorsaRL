"""Visualize saved frame-stack datasets (.npz) from Assetto Corsa.

Usage:
    python assetto-corsa-rl\scripts\ac\vis_image_dataset.py --input-dir datasets/ac_images

Controls (when window active):
 - n : next sample
 - b : previous sample
 - space : toggle play/pause (anim mode)
 - m : toggle view mode (anim / montage)
 - > : speed up (decrease delay)
 - < : slow down (increase delay)
 - s : save current visualization (PNG)
 - q or ESC : quit

Displays either an animated single-frame playback (`anim`) or a montage of the stacked frames (`montage`).
"""

from pathlib import Path
import argparse
import sys
import time
import math

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Visualize frame-stack dataset (.npz)")
    p.add_argument(
        "--input-dir", type=Path, required=True, help="Directory with .npz stacks"
    )
    p.add_argument(
        "--pattern", type=str, default="*.npz", help="Glob pattern to find stacks"
    )
    p.add_argument(
        "--delay", type=float, default=0.08, help="Frame playback delay in seconds"
    )
    p.add_argument("--scale", type=float, default=1.0, help="Display scale factor")
    p.add_argument("--start", type=int, default=0, help="Starting sample index")
    p.add_argument(
        "--view-mode",
        choices=["anim", "montage"],
        default="anim",
        help="Initial view mode",
    )
    p.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="If set, will save visualizations when pressing 's'",
    )
    return p.parse_args()


def list_files(input_dir: Path, pattern: str):
    files = sorted([p for p in input_dir.glob(pattern) if p.is_file()])
    return files


def load_stack(path: Path) -> np.ndarray:
    try:
        d = np.load(str(path))
        if "stack" in d:
            stack = d["stack"]
        else:
            # try first array
            keys = [k for k in d.files]
            stack = d[keys[0]]
        stack = np.asarray(stack)
        if stack.ndim != 3:
            raise ValueError(f"Expected stack with shape (F, H, W), got {stack.shape}")
        return stack.astype(np.uint8)
    except Exception as e:
        raise RuntimeError(f"Failed to load {path}: {e}")


def make_montage(stack: np.ndarray) -> np.ndarray:
    # stack: (F, H, W), uint8
    F, H, W = stack.shape
    cols = math.ceil(math.sqrt(F))
    rows = math.ceil(F / cols)
    # pad with zeros frames
    pad = cols * rows - F
    if pad > 0:
        pad_frames = np.zeros((pad, H, W), dtype=np.uint8)
        stack = np.concatenate([stack, pad_frames], axis=0)
    tiles = []
    for r in range(rows):
        row_frames = [stack[r * cols + c] for c in range(cols)]
        row_img = cv2.hconcat(row_frames)
        tiles.append(row_img)
    montage = cv2.vconcat(tiles)
    return montage


def to_bgr(img: np.ndarray) -> np.ndarray:
    # img is grayscale HxW
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def overlay_text(img: np.ndarray, text: str) -> None:
    cv2.putText(
        img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
    )


def main():
    args = parse_args()
    files = list_files(args.input_dir, args.pattern)
    if len(files) == 0:
        print(f"No files found in {args.input_dir} matching {args.pattern}")
        sys.exit(1)

    idx = max(0, min(args.start, len(files) - 1))

    window_name = "AC Dataset Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    playing = False
    view_mode = args.view_mode
    delay = args.delay
    frame_idx = 0
    last_time = time.time()

    while True:
        path = files[idx]
        try:
            stack = load_stack(path)
        except Exception as e:
            print(e)
            # skip to next
            idx = (idx + 1) % len(files)
            continue

        F, H, W = stack.shape

        if view_mode == "montage":
            montage = make_montage(stack)
            display_img = to_bgr(montage)
            title = f"{idx+1}/{len(files)} {path.name} [montage]"
        else:
            # anim mode: show single frame at frame_idx
            frame_idx = frame_idx % F
            frame = stack[frame_idx]
            display_img = to_bgr(frame)
            title = f"{idx+1}/{len(files)} {path.name} [frame {frame_idx+1}/{F}]"

        # overlay filename and instructions
        overlay = display_img.copy()
        overlay_text(overlay, title)
        overlay_text(
            overlay, "n:next  b:prev  space:play/pause  m:toggle view  s:save  q:quit"
        )

        # apply scaling
        if args.scale != 1.0:
            h = int(overlay.shape[0] * args.scale)
            w = int(overlay.shape[1] * args.scale)
            overlay = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_LINEAR)

        cv2.imshow(window_name, overlay)

        # manage playback timing
        key = cv2.waitKey(int(max(1, delay * 1000))) & 0xFF
        if key != 0xFF:
            # handle key
            if key == ord("q") or key == 27:  # esc
                break
            elif key == ord("n"):
                idx = (idx + 1) % len(files)
                frame_idx = 0
                playing = False
            elif key == ord("b"):
                idx = (idx - 1) % len(files)
                frame_idx = 0
                playing = False
            elif key == ord(" "):
                playing = not playing
            elif key == ord("m"):
                view_mode = "montage" if view_mode == "anim" else "anim"
                frame_idx = 0
            elif key == ord(">"):
                delay = max(0.001, delay * 0.5)
            elif key == ord("<"):
                delay = delay * 1.5
            elif key == ord("s"):
                save_dir = args.save_dir or args.input_dir
                save_dir.mkdir(parents=True, exist_ok=True)
                if view_mode == "montage":
                    out = make_montage(stack)
                else:
                    out = stack[frame_idx]
                out_path = save_dir / f"viz_{idx+1:06d}_{path.stem}.png"
                cv2.imwrite(str(out_path), out)
                print(f"Saved visualization: {out_path}")

        # advance frame if playing and in anim mode
        if playing and view_mode == "anim":
            now = time.time()
            if now - last_time >= delay:
                frame_idx = (frame_idx + 1) % F
                last_time = now

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
