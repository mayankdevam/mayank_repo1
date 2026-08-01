import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Correctly specify the path to tourism.csv from the execution directory (/content/)
df = pd.read_csv("tourism_project/data/tourism.csv")

# Drop CustomerID as it's a unique identifier and not a feature
df.drop(columns=["CustomerID"], inplace=True)

X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# stratify=y keeps the (imbalanced) target ratio consistent across splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define the directory where the split files should be saved
output_dir = "tourism_project/model_building"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Save the split data to the specified output directory
Xtrain.to_csv(os.path.join(output_dir, "Xtrain.csv"), index=False)
Xtest.to_csv(os.path.join(output_dir, "Xtest.csv"), index=False)
ytrain.to_csv(os.path.join(output_dir, "ytrain.csv"), index=False)
ytest.to_csv(os.path.join(output_dir, "ytest.csv"), index=False)

print("Data prepared: train/test splits written to `tourism_project/model_building/`.")
