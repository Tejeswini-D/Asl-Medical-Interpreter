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

# Camera
cap = cv2.VideoCapture(0)

# Label input
label = input("Enter label (yes or no): ").strip().lower()

# Store data from this session
data = []

print("Recording... Press Q to stop")

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
            row.append(label)
            data.append(row)

    cv2.imshow("Recording YES / NO", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
hands.close()
cv2.destroyAllWindows()

# Column names
columns = []
for i in range(21):
    columns.extend([f"x{i}", f"y{i}", f"z{i}"])
columns.append("label")

# New data
df_new = pd.DataFrame(data, columns=columns)

# File path
file_path = "yes_no_mediapipe.csv"

# Append safely
if os.path.exists(file_path):
    df_existing = pd.read_csv(file_path)
    df_final = pd.concat([df_existing, df_new], ignore_index=True)
else:
    df_final = df_new

# Save
df_final.to_csv(file_path, index=False)

print("✅ Data appended to yes_no_mediapipe.csv")
print(df_final["label"].value_counts())
