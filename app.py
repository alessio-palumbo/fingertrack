import cv2
import mediapipe as mp

# import requests

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Hikari endpoint
HIKARI_URL = "http://localhost:8080/gesture"


# Simple gesture mapping function
def detect_gesture(hand_landmarks):
    # Example: check if all fingers are up -> 'open_palm'
    fingers_up = []
    tips = [8, 12, 16, 20]  # fingertip landmark indices
    for tip_id in tips:
        # If the tip is above the joint (smaller y value), that means the finger is extended/up.
        if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
            fingers_up.append(1)
        else:
            fingers_up.append(0)
    if sum(fingers_up) == 4:
        return "open_palm"
    elif sum(fingers_up) == 0:
        return "fist"
    return "unknown"


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            gesture = detect_gesture(hand_landmarks)
            if gesture != "unknown":
                # Send event to Hikari
                # requests.post(HIKARI_URL, json={"gesture": gesture})
                print(f"Gesture was {gesture}")

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Gestures", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
