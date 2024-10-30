import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Load your Excel data
df = pd.read_excel("./inference_results.xlsx")

# Assuming 'Image ID' contains actual PIL images or you can load them
# If needed, adjust this step to ensure 'Image ID' is a PIL image

# Now, we can plot the first image from the DataFrame
first_image = df["Image ID"].iloc[0]

# Verify it's a PIL image, if not, handle the conversion
if isinstance(first_image, Image.Image):
    # Plot the first image using matplotlib
    plt.imshow(first_image)
    plt.axis('off')  # Turn off axis labels for a cleaner view
    plt.show()
else:
    print("The first entry in 'Image ID' is not a valid PIL image.")
