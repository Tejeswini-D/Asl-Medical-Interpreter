import pandas as pd

df = pd.read_csv("yes_no_mediapipe.csv")
print(df['label'].value_counts())
