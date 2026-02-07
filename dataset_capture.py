import cv2
import mediapipe as mp
import csv
import os
csv_file="gesture_dataset.csv"
if not os.path.exists(csv_file):
    with open(csv_file,mode='w',newline='') as f:
        writer=csv.writer(f)
        header=[]
        for i in range(21*2):
            header+=[f'hand{i}_x',f'hand{i}_y',f'hand{i}_z']
        header+=['label']
        writer.writerow(header)
mp_hands=mp.solutions.hands
mp_drawing=mp.solutions.drawing_utils
hands=mp_hands.Hands(max_num_hands=2,min_detection_confidence=0.5,min_tracking_confidence=0.5)
cap=cv2.VideoCapture(0)
current_label=input("enter label for this recording: ")
print("recording hand landmarks..")
while True:
    ret,frame=cap.read()
    if not ret:
        break
    frame=cv2.flip(frame,1)
    rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    hand_results=hands.process(rgb_frame)
    row=[]
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                row+=[lm.x,lm.y,lm.z]
        while len(row)<21*2*3:
            row+=[0,0,0]
    else:
        row+=[0]*(21*2*3)
    row+=[current_label]
    with open(csv_file,mode='a',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(row)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame,hand_landmarks,mp_hands.HAND_CONNECTIONS)
    cv2.imshow("HAND TRACKING",frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()