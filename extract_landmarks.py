import cv2
import mediapipe as mp
import pandas as pd
import os
mp_hands=mp.solutions.hands
hands=mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
data=[]
labels=["i","you","she"]
base_dir="wlasl_video"
for label in labels:
    folder=os.path.join(base_dir,label)
    print(f"Processing {label}...")
    for video in os.listdir(folder):
        video_path=os.path.join(folder,video)
        cap=cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret,frame=cap.read()
            if not ret:
                break
            frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            result=hands.process(frame_rgb)
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    row=[]
                    for lm in hand_landmarks.landmark:
                        row.extend([lm.x,lm.y,lm.z])
                    row.append(label)
                    data.append(row)
        cap.release()
hands.close()
columns=[]
for i in range(21):
    columns.extend([f"x{i}",f"y{i}",f"z{i}"])
columns.append("label")
df=pd.DataFrame(data,columns=columns)
df.to_csv("asl_pronouns_dataset.csv",index=False)
print("dataset saved")