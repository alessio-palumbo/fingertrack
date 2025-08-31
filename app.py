import cv2

from gesture_detector import GestureDetector  # assuming class is in gesture_detector.py


def main():
    cap = cv2.VideoCapture(0)
    detector = GestureDetector()

    gesture_history = []
    history_size = 5

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror frame
        mirrored_frame = cv2.flip(frame, 1)

        hands_data = detector.detect(mirrored_frame)
        for landmarks, hand_label in hands_data:
            fingers = detector.fingers_up(landmarks, hand_label)
            gesture = detector.classify_gesture(fingers)

            # Smoothing / Debouncing
            gesture_history.append(gesture)
            if len(gesture_history) > history_size:
                gesture_history.pop(0)

            stable_gesture = max(set(gesture_history), key=gesture_history.count)

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
