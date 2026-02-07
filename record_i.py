import cv2
import mediapipe as mp
import pandas as pd
import os

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

data = []

print("Recording label: I")
print("Press Q to stop")

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
            row.append("i")   # 🔥 fixed label
            data.append(row)

    cv2.imshow("Recording I", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()

# Columns
columns = []
for i in range(21):
    columns.extend([f"x{i}", f"y{i}", f"z{i}"])
columns.append("label")

df_new = pd.DataFrame(data, columns=columns)

file_path = "asl_pronouns_dataset.csv"

if os.path.exists(file_path):
    df_existing = pd.read_csv(file_path)
    df_final = pd.concat([df_existing, df_new], ignore_index=True)
else:
    df_final = df_new

df_final.to_csv(file_path, index=False)

print("✅ I samples added")
print(df_final["label"].value_counts())
