from datasets import Dataset, Features, Value, Image, DatasetDict
import pandas as pd

df = pd.read_csv("./dataset_sa.csv",index_col=0)
df.drop(columns=["new_lines"],inplace=True)
train_dataset = df[df["split"] == "Train"]
test_dataset = df[df["split"] == "Test"]
valid_dataset = df[df["split"] == "Valid"]
print(train_dataset.head())

# Convert the DataFrame to a Hugging Face Dataset with image features
# image_id,case_id,question,answer,ans_len,mode,split
features = Features(
    {
        "image_id": Image(),
        "case_id": Value("string"),
        "question": Value("string"),
        "answer": Value("string"),
        "ans_len": Value("int32"),
        "mode": Value("string"),
        "split": Value("string"),
    }
)
train_dataset = Dataset.from_pandas(train_dataset,features=features)
test_dataset = Dataset.from_pandas(test_dataset,features=features)
valid_dataset = Dataset.from_pandas(valid_dataset,features=features)

dataset_dict = DatasetDict(
    {"Train": train_dataset, "Test": test_dataset, "Valid": valid_dataset}
)

# Now you can push the dataset to the Hugging Face Hub or work with it locally
dataset_dict.push_to_hub("adishourya/MEDPIX-ShortQA")

