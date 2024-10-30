import json
import pandas as pd

with open("./ours_qa_pairs.json") as f:
    qa = json.load(f)

qa_df = pd.DataFrame(qa)
qa_df.to_excel("QA_DATASET.xlsx")
