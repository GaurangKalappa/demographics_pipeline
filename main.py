#!/usr/bin/env python3
# ============================================================
#  main.py  –  Entry point for the Demographics Pipeline
#
#  Usage examples:
#    python main.py                          # webcam (device 0)
#    python main.py --source video.mp4       # video file
#    python main.py --source rtsp://IP/stream
#    python main.py --fps 15 --no-display
#    python main.py --tripwire 0.0,0.6,1.0,0.6
# ============================================================

import argparse
import sys
import os

# Ensure project root is on the path when run from any cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from pipeline import DemographicsPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hierarchical Demographics Pipeline (age + gender from video)")

    parser.add_argument(
        "--source", type=str, default=None,
        help="Video source: 0 (webcam), path/to/video.mp4, rtsp://… "
             f"(default from config: {config.VIDEO_SOURCE})")

    parser.add_argument(
        "--fps", type=int, default=None,
        help=f"Target FPS for capture (default from config: {config.FPS_TARGET})")

    parser.add_argument(
        "--no-display", action="store_true",
        help="Disable the live OpenCV window (headless mode)")

    parser.add_argument(
        "--tripwire", type=str, default=None,
        help="Tripwire as 'x1,y1,x2,y2' in normalised [0,1] coords "
             "(e.g. '0.0,0.55,1.0,0.55')")

    parser.add_argument(
        "--person-conf", type=float, default=None,
        help=f"Person detection confidence threshold "
             f"(default: {config.PERSON_CONF_THRESH})")

    parser.add_argument(
        "--log", type=str, default=None,
        help=f"Path to output JSONL log file (default: {config.LOG_PATH})")

    return parser.parse_args()


def apply_cli_overrides(args):
    """Patch config values with CLI arguments where provided."""
    if args.source is not None:
        # Convert numeric string to int for webcam device
        try:
            config.VIDEO_SOURCE = int(args.source)
        except ValueError:
            config.VIDEO_SOURCE = args.source

    if args.fps is not None:
        config.FPS_TARGET = args.fps

    if args.no_display:
        config.DISPLAY_WINDOW = False

    if args.tripwire is not None:
        try:
            x1, y1, x2, y2 = [float(v) for v in args.tripwire.split(",")]
            config.TRIPWIRE_START = (x1, y1)
            config.TRIPWIRE_END   = (x2, y2)
        except ValueError:
            print("[main] ✖  --tripwire must be 'x1,y1,x2,y2' floats. Using default.")

    if args.person_conf is not None:
        config.PERSON_CONF_THRESH = args.person_conf

    if args.log is not None:
        config.LOG_PATH = args.log


def print_startup_banner():
    print("=" * 60)
    print("  Demographic Extraction Pipeline")
    print("  Age Estimation + Gender Identification")
    print("=" * 60)
    print(f"  Source       : {config.VIDEO_SOURCE}")
    print(f"  FPS target   : {config.FPS_TARGET}")
    print(f"  Tripwire     : {config.TRIPWIRE_START} → {config.TRIPWIRE_END}")
    print(f"  Buffer size  : {config.BUFFER_SIZE} frames")
    print(f"  Log path     : {config.LOG_PATH}")
    print(f"  Display      : {'ON' if config.DISPLAY_WINDOW else 'OFF (headless)'}")
    print("=" * 60 + "\n")


def main():
    args = parse_args()
    apply_cli_overrides(args)
    print_startup_banner()

    pipeline = DemographicsPipeline()
    pipeline.run(config.VIDEO_SOURCE)


if __name__ == "__main__":
    main()