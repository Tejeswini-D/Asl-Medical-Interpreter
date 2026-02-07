import pandas as pd

df_yesno = pd.read_csv("yes_no_mediapipe.csv")
df_pronouns = pd.read_csv("asl_pronouns_dataset.csv")

# Merge labels: she + you → person
df_pronouns["label"] = df_pronouns["label"].replace({
    "she": "they",
    "you": "they"
})

# Sanity check
print("Pronoun labels after merge:", df_pronouns["label"].unique())

# Columns must match
assert list(df_yesno.columns) == list(df_pronouns.columns)

# Combine datasets
df_final = pd.concat([df_yesno, df_pronouns], ignore_index=True)
df_final = df_final.sample(frac=1).reset_index(drop=True)

df_final.to_csv("final_asl_dataset.csv", index=False)

print("Final labels:", df_final["label"].value_counts())
