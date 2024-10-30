from datasets import Dataset, Features , Value , Image , DatasetDict, load_dataset
from numpy import imag
import pandas as pd
import matplotlib.pyplot as plt
import io
from PIL import Image as PILImage
import ast

df_test = pd.read_csv("./test.csv")
df_valid = pd.read_csv("./validation.csv")


def plot_one(df):
    img0 = df.iloc[0]["image"]

    # Convert the string back to a dictionary
    image_data_dict = ast.literal_eval(img0)  # Replace this with the actual output

    # Extract the image bytes
    image_bytes = image_data_dict['bytes']

    # Use PIL to open the image from bytes
    image = PILImage.open(io.BytesIO(image_bytes))

    # Plot the image
    plt.imshow(image)
    plt.axis('off')  # Hide axes for better display
    plt.show()

df_valid["image"] = df_valid["image"].apply(lambda x : ast.literal_eval(x)["bytes"])
df_test["image"] = df_test["image"].apply(lambda x : ast.literal_eval(x)["bytes"])


features = Features(
    {
        "image_id": Value("string"),
        "image": Image(decode=False),
        "question": Value("string"),
        "answer": Value("string"),
    }
)
df_test = Dataset.from_pandas(df_test,features=features)
df_valid = Dataset.from_pandas(df_valid,features=features)
dataset_dict = DatasetDict(
    {"Vaild": df_valid,
     "Test":df_test} 
)

dataset_dict.push_to_hub("adishourya/ROCO-QA")
#
