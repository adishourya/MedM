import numpy as np
import pandas as pd
import plotly.express as px

# Load your data
df = pd.read_excel("./description_dict.xlsx", index_col=0)

# Convert all columns to strings to handle mixed data types
df = df.astype(str)

# Group by 'modality', 'plane', and 'location' (removing 'location_category')
grouped_data = df.groupby(['modality', 'plane', 'location']).size().reset_index(name='size')

# Create the treemap using Plotly
fig = px.treemap(grouped_data,
                 path=['modality', 'plane', 'location'],  # Define hierarchy
                 values='size',  # Size of each block
                 color='size',   # Color by size
                 color_continuous_scale='Blues')  # Choose a new color scheme

# Customize the font and layout
fig.update_layout(
    title='Medpix Treemap',
    font=dict(
        # family="Courier New, monospace",  # Change to your preferred font
        family="Geist Mono",  # Change to your preferred font
        size=16,  # Font size
        color="black"  # Font color
    ),
    paper_bgcolor='white',  # Background color
    margin=dict(t=50, l=25, r=25, b=25)  # Adjust margins for better layout
)

# Update the font for the text inside the treemap boxes
fig.update_traces(
    insidetextfont=dict(
        family="Geist Mono",  # Font for labels inside the boxes
        size=14,  # Font size for labels
        color="black"  # Font color for labels
    )
)

# Show the treemap
fig.show()
