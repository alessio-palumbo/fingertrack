import json

import cv2

from consumer import get_consumers_from_args
from fingers_detector import FingersDetector


def main():
    cap = cv2.VideoCapture(0)
    consumers = get_consumers_from_args()
    detector = FingersDetector(consumers=consumers)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror frame for more natural video output.
        mirrored_frame = cv2.flip(frame, 1)

        hands_data = detector.detect(mirrored_frame)
        for landmarks, hand_label in hands_data:
            detector.process_hand(mirrored_frame, landmarks, hand_label)

    cap.release()
    for consumer in consumers:
        if hasattr(consumer, "close"):
            consumer.close()


if __name__ == "__main__":
    main()
