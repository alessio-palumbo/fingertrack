import signal

import cv2

from consumer import get_consumers_from_args
from fingers_detector import FingersDetector
from hand_tracker import HandTracker


def main():
    cap = cv2.VideoCapture(0)
    consumers = get_consumers_from_args()
    tracker = HandTracker()
    detector = FingersDetector(consumers=consumers)

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
            hands_data = tracker.detect(mirrored_frame)
            detector.process_hands(mirrored_frame, hands_data)

    finally:
        cap.release()
        for consumer in consumers:
            consumer.close()


if __name__ == "__main__":
    main()
