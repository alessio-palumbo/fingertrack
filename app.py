import signal
import sys

import cv2

from consumer import get_consumers_from_args
from hand import HandEngine

# Silently exit when pipe is closed.
signal.signal(signal.SIGPIPE, signal.SIG_DFL)


def main():
    cap = cv2.VideoCapture(0)
    consumers = get_consumers_from_args()
    hand_engine = HandEngine(consumers=consumers)

    running = True

    # Signal handler to stop gracefully
    def stop_gracefully(signum, _):
        nonlocal running
        print(f"\nSignal {signum} received. Exiting gracefully...")
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

    finally:
        cap.release()
        cv2.destroyAllWindows()
        sys.exit(0)


if __name__ == "__main__":
    main()
