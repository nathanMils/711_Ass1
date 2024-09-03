# %%
# Import required libraries

# Pandas for data preparation and Numpty for DP logic
import pandas as pd
import numpy as np

# %%
# Load RawData
rawData = pd.read_csv("../Data/Almond.csv")

# %% [markdown]
# # Data Preparation
# ## Multiple Imputation
# Since there is a large amount of missing data with:
# - 31% missing Length
# - 34% missing Width
# - 36% missing Thickness
# Simple median or mean imputation will not due, hence we shall use multiple imputation for these 3 attributes.
# ## Derived Attributes
# The attributes of Aspect Ratio and Eccentricity are derived from length and width and are missing where either length or width is missing and so we can calculate these attributes using the new imputed values.
# ## Roundness Exception
# Roundness is different from the other derived attributes in that it is calculated from both Area and Length. Since area is obviously affected by the profile taken of the almond (Top/Side/Front) we cannot simply interpret roundness without 

# %%
# Retrieve Length, Width and Thickness for imputation
p_LWT = rawData[['Length (major axis)','Width (minor axis)','Thickness (depth)','Area']].copy()

p_LWT['Area'] = np.where(p_LWT['Length (major axis)'].notna(),
                          p_LWT['Area'],
                          np.nan)

# %%
# Import Sklearn for multiple imputation
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
# For binary one-hot encoding
from sklearn.preprocessing import LabelEncoder
# For testing and training split
from sklearn.model_selection import train_test_split

# %%
# Use multiple imputation using sklearn
imputer = IterativeImputer(max_iter=10, random_state=0)
d_LWT_imputed = pd.DataFrame(imputer.fit_transform(p_LWT), columns=p_LWT.columns)

# %%
# Calculate Roundness using the imputed Area when there is length
d_LWT_imputed['Roundness'] = 4 * d_LWT_imputed['Area'] / (np.pi * d_LWT_imputed['Length (major axis)']**2)

# %%
# Remove irrelavent features
p_proc = rawData.drop('Id',axis=1)
# Use imputed data to calculate derived features
p_proc[['Length (major axis)','Width (minor axis)','Thickness (depth)','Roundness']] = d_LWT_imputed[['Length (major axis)','Width (minor axis)','Thickness (depth)','Roundness']]
p_proc['Aspect Ratio'] = p_proc['Length (major axis)']/p_proc['Width (minor axis)']
p_proc['Eccentricity'] = (1 - (p_proc['Width (minor axis)']/p_proc['Length (major axis)'])**2) ** 0.5

# %%
# Normalization
p_norm = p_proc[['Length (major axis)','Width (minor axis)','Thickness (depth)','Area','Perimeter','Roundness','Solidity','Compactness','Aspect Ratio','Eccentricity','Extent','Convex hull(convex area)']]
p_norm = (p_norm - p_norm.mean()) / p_norm.std()

# %%
# Binary One Hot Encoding
labeler = LabelEncoder()

# %%
# Input
X = p_norm
# Target
Y = p_proc['Type']

# %%
# Import libraries for NN
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# %%
X_tensor = torch.tensor(X.values, dtype=torch.float32)
y_tensor = torch.tensor(labeler.fit_transform(Y), dtype=torch.long)

# %%
# Splitting Dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# %% [markdown]
# # Neural Network Definition

# %%
class NathansWeirdNN(nn.Module):
    def __init__(self):
        super(NathansWeirdNN, self).__init__()
        self.fc1 = nn.Linear(X_tensor.shape[1], 64)  # First hidden layer
        self.fc2 = nn.Linear(64, 32)                 # Second hidden layer
        self.fc3 = nn.Linear(32, 3)                  # Output layer (3 classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# %%
model = NathansWeirdNN()
criterion = nn.CrossEntropyLoss()  # For classification tasks
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# %% [markdown]
# ## Training Algorithm

# %%
num_epochs = 100  # Number of epochs to train

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode

    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train.long())  # Ensure y_train is of type LongTensor for classification

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# %%



