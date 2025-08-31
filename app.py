import cv2

from gesture_detector import GestureDetector  # assuming class is in gesture_detector.py


def main():
    cap = cv2.VideoCapture(0)
    detector = GestureDetector()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        hands_data = detector.detect(frame)
        for landmarks, hand_label in hands_data:
            fingers = detector.fingers_up(landmarks, hand_label)
            gesture = detector.classify_gesture(fingers)

            # Draw landmarks
            detector.mp_draw.draw_landmarks(
                frame, landmarks, detector.mp_hands.HAND_CONNECTIONS
            )

            # Display gesture
            cv2.putText(
                frame,
                f"{hand_label} - {gesture}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

        cv2.imshow("Gesture Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
