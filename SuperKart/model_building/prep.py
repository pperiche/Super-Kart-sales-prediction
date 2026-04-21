# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split

# for hugging face space authentication to upload files
from huggingface_hub import HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/PratzPrathibha/Super-kart-sales-prediction/SuperKart.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop the unique identifier
df.drop(columns=['Product_Id', 'Store_Id', 'count'], inplace=True)

# Fix dirty value
df["Product_Sugar_Content"] = df["Product_Sugar_Content"].replace("reg","Regular")

# Ordinal encoding
size_map = {"Small":0,"Medium":1,"High":2}
df["Store_Size"] = df["Store_Size"].map(size_map)

city_map = {"Tier 3":0,"Tier 2":1,"Tier 1":2}
df["Store_Location_City_Type"] = df["Store_Location_City_Type"].map(city_map)

# One-hot encoding
df = pd.get_dummies(df, columns=[
    "Product_Sugar_Content",
    "Product_Type",
    "Store_Type"
], drop_first=True)

# Handle skewness using log1p
df["Product_Allocated_Area"] = np.log1p(df["Product_Allocated_Area"])

# Transform target variable
df["Product_Store_Sales_Total"] = np.log1p(df["Product_Store_Sales_Total"])

target_col = 'Product_Store_Sales_Total'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# save a copy of train and test data
train = Xtrain.copy()
train["Product_Store_Sales_Total"] = ytrain

test = Xtest.copy()
test["Product_Store_Sales_Total"] = ytest

# Creating a csv file from test and train data.
Xtrain.to_csv("SuperKart/data/Xtrain.csv",index=False)
Xtest.to_csv("SuperKart/data/Xtest.csv",index=False)
ytrain.to_csv("SuperKart/data/ytrain.csv",index=False)
ytest.to_csv("SuperKart/data/ytest.csv",index=False)

files = [
"SuperKart/data/Xtrain.csv",
"SuperKart/data/Xtest.csv",
"SuperKart/data/ytrain.csv",
"SuperKart/data/ytest.csv"
]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="PratzPrathibha/Super-kart-sales-prediction",
        repo_type="dataset",
    )
