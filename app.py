import json

import cv2

from consumer import get_consumer_from_args
from fingers_detector import FingersDetector


def main():
    cap = cv2.VideoCapture(0)
    detector = FingersDetector()
    consumer = get_consumer_from_args()

    # Fingers change tracking
    last_fingers = None
    # Track fingers history for smoothing
    fingers_history = []
    history_size = 5

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror frame for more natural video output.
        mirrored_frame = cv2.flip(frame, 1)

        hands_data = detector.detect(mirrored_frame)
        for landmarks, hand_label in hands_data:
            fingers = detector.fingers_up(landmarks, hand_label)

            # Smoothing / Debouncing
            fingers_history.append(tuple(fingers))
            if len(fingers_history) > history_size:
                fingers_history.pop(0)

            stable_fingers = max(set(fingers_history), key=fingers_history.count)
            if stable_fingers != last_fingers:
                consumer.consume(stable_fingers)
                last_fingers = stable_fingers

            stable_gesture = detector.classify_gesture(stable_fingers)

            # Draw landmarks
            detector.mp_draw.draw_landmarks(
                mirrored_frame, landmarks, detector.mp_hands.HAND_CONNECTIONS
            )

            # Display gesture
            cv2.putText(
                mirrored_frame,
                f"{hand_label} - {stable_gesture}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
            )

        cv2.imshow("Gesture Detector", mirrored_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
