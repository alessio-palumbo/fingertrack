import argparse
import signal
import sys

import cv2

from consumer import BaseConsumer, HttpConsumer, OpenCVWindowConsumer, StdoutConsumer
from hand import HandEngine

# Ignore SIGPIPE on Unix (not available on Windows)
if hasattr(signal, "SIGPIPE"):
    signal.signal(signal.SIGPIPE, signal.SIG_IGN)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for consumers and hand engine."""

    parser = argparse.ArgumentParser()

    # --- Consumer args ---
    parser.add_argument(
        "--consumer",
        choices=["stdout", "http"],
        default="stdout",
        help="Select output consumer (default: stdout)",
    )
    parser.add_argument("--url", help="URL for http consumer")
    parser.add_argument(
        "--show-window", action="store_true", help="Display OpenCV feed"
    )

    # --- HandEngine args (optional, only if HandEngine wants to parse CLI) ---
    parser.add_argument(
        "--frame-skip", type=int, default=1, help="Process every Nth frame"
    )
    parser.add_argument(
        "--buffer-size", type=int, default=5, help="Number of frames to buffer"
    )
    parser.add_argument(
        "--gesture-threshold",
        type=float,
        default=0.1,
        help="Gesture detection threshold",
    )

    args = parser.parse_args()
    return args


def get_consumers(args: argparse.Namespace) -> list[BaseConsumer]:
    """Create consumer instances based on parsed args."""

    consumers = []
    if args.show_window:
        consumers.append(OpenCVWindowConsumer())
    if args.consumer == "http":
        consumers.append(HttpConsumer(args.url))
    else:
        consumers.append(StdoutConsumer())
    return consumers


def main():
    """
    Initializes the hand detection and gesture recognition engine, sets up
    consumers for event handling, and starts the main processing loop.

    Features:
    - Detects hand landmarks using MediaPipe.
    - Recognizes gestures (swipes, finger patterns, etc.).
    - Dispatches events to one or more consumers, such as:
        - StdoutConsumer: logs events to console
        - HttpConsumer: sends events to a remote endpoint
        - OpenCVWindowConsumer: visualizes the camera feed in a window
    - Supports configurable options:
        - --consumer: type of consumer to use
        - --url: URL for HTTP consumer
        - --show-window: display live camera feed
        - --frame-skip: number of frames to skip between processing
        - --buffer-size: number of frames to buffer for detection smoothing
        - --gesture-threshold: minimum movement required to detect a gesture

    Example usage:
        python main.py --consumer stdout --show-window --gesture-buffer 5

    Raises:
        SystemExit: if required resources (camera, config, etc.) are unavailable.
    """
    cap = cv2.VideoCapture(0)
    args = parse_args()
    hand_engine = HandEngine(
        frame_skip=args.frame_skip,
        buffer_size=args.buffer_size,
        gesture_threshold=args.gesture_threshold,
        consumers=get_consumers(args),
    )

    running = True

    # Signal handler to stop gracefully
    def stop_gracefully(signum, _):
        nonlocal running
        try:
            print(f"\nSignal {signum} received. Exiting gracefully...")
        except BrokenPipeError:
            # stdout/stderr already closed, just continue shutdown
            pass
        running = False

    signal.signal(signal.SIGINT, stop_gracefully)
    signal.signal(signal.SIGTERM, stop_gracefully)

    try:
        while cap.isOpened() and running:
            ret, frame = cap.read()
            if not ret:
                break

            # Mirror frame for more natural video output.
            mirrored_frame = cv2.flip(frame, 1)
            hand_engine.process_frame(mirrored_frame)

    except BrokenPipeError:
        # Handle case where the stdout pipe is closed early
        running = False

    finally:
        cap.release()
        cv2.destroyAllWindows()
        sys.exit(0)


if __name__ == "__main__":
    main()
