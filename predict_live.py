import cv2
import mediapipe as mp
import pandas as pd
import joblib
import numpy as np

# Load model
model = joblib.load("asl_model.pkl")
labels = model.classes_

# MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

columns = []
for i in range(21):
    columns.extend([f"x{i}", f"y{i}", f"z{i}"])

print("Live prediction started. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            row = []
            for lm in hand_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])

            X = pd.DataFrame([row], columns=columns)
            probs = model.predict_proba(X)[0]
            pred = labels[np.argmax(probs)]
            conf = np.max(probs)

            cv2.putText(
                frame,
                f"{pred.upper()} ({conf:.2f})",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3
            )

    cv2.imshow("ASL Medical Interpreter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()
