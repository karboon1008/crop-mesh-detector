#!/usr/bin/env python3
"""Collects "no leaf" photos with a webcam, for retraining the leaf detector so it stops
boxing people, faces and the room (src/detection/train_detector.py --negatives-dir).

The best negatives are the ones the demo will actually see: run this on the demo laptop,
in the demo room, and move around in front of the camera with NO leaves or plants in
view. Include faces close up and far away, hands, arms, several people, clothes of
different colours (especially green), the desk, walls, windows and screens.

    python scripts/capture_negatives.py                       # 300 frames, one every 0.5 s
    python scripts/capture_negatives.py --count 500 --interval 0.3 --out data/negatives/room2

A preview window opens; capture starts after a 5 s countdown. Press q to stop early.
Look through the folder afterwards and delete any frame that has a leaf or plant in it.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/negatives/webcam", help="Folder to save the frames in")
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--interval", type=float, default=0.5, help="Seconds between saved frames")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--countdown", type=float, default=5.0)
    args = parser.parse_args()

    import cv2

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.device_index)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.device_index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    start = time.time()
    saved, last = 0, 0.0
    try:
        while saved < args.count:
            ok, frame = cap.read()
            if not ok:
                raise SystemExit("Failed to read from the camera")
            now = time.time()
            waiting = args.countdown - (now - start)
            display = frame.copy()
            if waiting > 0:
                text = f"No leaves in view! Starting in {waiting:.0f}s"
            else:
                text = f"Saved {saved}/{args.count} - keep moving, NO leaves (q to stop)"
                if now - last >= args.interval:
                    cv2.imwrite(str(out / f"neg-{stamp}-{saved:04d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    saved += 1
                    last = now
            cv2.putText(display, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.imshow("capture negatives", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print(f"Saved {saved} frames to {out}/. Delete any that show a leaf or plant, then retrain with "
          f"--negatives-dir {out.parent}")


if __name__ == "__main__":
    main()
