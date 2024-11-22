





# import streamlit as st
# from fastai.vision import *
# from pathlib import Path
# import subprocess
# import os
# import numpy as np  # Ensure numpy is imported

# # Set up title and description for the app
# st.title('Intel Image Classification with FastAI')
# st.write("This Streamlit app demonstrates the process of training an image classifier using FastAI.")

# # Define the dataset path
# dataset_url = "https://github.com/getnetbogale27/Image-classification.git"
# local_repo_path = Path("image_classification_repo")
# dataset_path = local_repo_path / "dataset/seg_train"

# # Clone or check the dataset
# if not dataset_path.exists():
#     st.info("Dataset not found locally. Cloning the GitHub repository...")
#     try:
#         subprocess.run(["git", "clone", dataset_url, local_repo_path.as_posix()], check=True)
#         st.success(f"Dataset cloned successfully to `{dataset_path}`.")
#     except Exception as e:
#         st.error(f"Failed to clone the repository. Error: {e}")
#         st.stop()
# else:
#     st.info(f"Dataset found locally at `{dataset_path}`.")

# # Load the dataset
# with st.expander("Dataset Information"):
#     st.write("Listing the classes in the dataset:")
#     try:
#         np.random.seed(40)  # Seed for reproducibility
        
#         # Create the data bunch
#         data = ImageDataBunch.from_folder(
#             dataset_path, 
#             train='.', 
#             valid_pct=0.2, 
#             ds_tfms=get_transforms(), 
#             size=224, 
#             num_workers=4
#         ).normalize(imagenet_stats)
        
#         # Display dataset information
#         st.write("Classes found: ", data.classes)
#         st.write("Number of classes: ", len(data.classes))
#     except Exception as e:
#         st.error(f"Error loading dataset: {e}")
#         st.stop()










import streamlit as st
import os
import pandas as pd

st.title("🎉 Welcome to Our Deployed App!")
st.write("Congratulations! Our Streamlit app has been successfully deployed. 🚀")

st.write("""
### Authors: Getnet B. (PhD Candidate)
""")

st.info("""
The goal of this mini-project is to design and implement an image classification model using Convolutional Neural Networks (CNNs). Students will learn to handle image data, preprocess it, train a CNN, improve its performance, and deploy the trained model on an open-source platform such as Hugging Face or Streamlit.
""")

# Define the base directories for training and test datasets
train_base_dir = "dataset/seg_train"
test_base_dir = "dataset/seg_test"

# List of categories
categories = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

# Function to count the number of images in a directory
def count_images_in_directory(directory):
    # Check if the directory exists
    if not os.path.exists(directory):
        st.error(f"Directory not found: {directory}")
        return 0
    return len([f for f in os.listdir(directory) if f.endswith(('jpg', 'jpeg', 'png', 'bmp'))])

# Create a list to store category names and image counts
category_names = []
train_counts = []
test_counts = []

# Loop through the categories and get the image counts for each category
for category in categories:
    # Construct the full paths to the category directories
    train_category_dir = os.path.join(train_base_dir, category)
    test_category_dir = os.path.join(test_base_dir, category)

    # Count the images in each category
    train_images_count = count_images_in_directory(train_category_dir)
    test_images_count = count_images_in_directory(test_category_dir)

    # Append the category names and image counts to the lists
    category_names.append(category.capitalize())
    train_counts.append(train_images_count)
    test_counts.append(test_images_count)

# Create a DataFrame for easy plotting
df = pd.DataFrame({
    'Category': category_names,
    'Training Images': train_counts,
    'Test Images': test_counts
})

# Create an expander to display the category counts and chart
with st.expander("🔍 Show Image Counts in Training and Test Datasets"):
    # Display the counts in a table
    st.write(df)

    # Plot the bar chart
    st.bar_chart(df.set_index('Category'))

