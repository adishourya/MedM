import numpy as np
import pandas as pd
from sqlalchemy.sql.expression import column
df = pd.read_excel("./description_dict.xlsx",index_col=0)
sep = " • "
df["subject"] = df["modality"] + sep + df["plane"] + sep + df["location"] + sep + df["location_category"]
df = df[["caption","subject"]]
df["patient_id"] = df.index.str.split("_").str[0]
df["image_id"] = df.index.str.split("_").str[1]

# 671 patients
# 90 percent of patients = 603.9
patients = dict.fromkeys(df["patient_id"],-1)
for p in patients.keys():
    patients[p] = np.random.choice(["Train","Valid","Test"],p=[0.9,0.05,0.05])
    
df["split"] = df["patient_id"].apply(lambda x : patients[x])
df.drop(columns=["image_id"],inplace=True)
df = df.drop_duplicates()

df.to_excel("dataset_description.xlsx",index=True)

