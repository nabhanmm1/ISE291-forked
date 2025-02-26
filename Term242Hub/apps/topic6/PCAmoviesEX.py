import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Streamlit App Setup
st.title("PCA Visualization App")
st.write("An interactive example demonstrating Principal Component Analysis (PCA) on customer movie preferences.")

# Data
st.subheader("Original Data")
data = np.array([
    [10, 2],
    [7, 5],
    [2, 8],
    [1, 10],
    [8, 3],
    [5, 5]
])
colors = ['red', 'green', 'blue', 'purple', 'orange', 'black']

# Scatter plot of original data
fig, ax = plt.subplots()
ax.scatter(data[:, 0], data[:, 1], c=colors)
ax.set_xlabel("Action Movies Watched")
ax.set_ylabel("Comedy Movies Watched")
ax.set_title("Color-Coded Customer Movie Preferences")
ax.grid(True)
st.pyplot(fig)

# Compute PCA
st.subheader("PCA Computation")
pca = PCA(n_components=2)
pca.fit(data)
principal_components = pca.transform(data)

# Scatter plot of PCA-transformed data
fig, ax = plt.subplots()
ax.scatter(principal_components[:, 0], principal_components[:, 1], c=colors)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("PCA of Customer Movie Preferences (Color-Coded)")
ax.grid(True)
st.pyplot(fig)

# Visualizing principal component vectors on the original data
st.subheader("Principal Components on Original Data")
fig, ax = plt.subplots()
ax.scatter(data[:, 0], data[:, 1], c=colors)
ax.set_xlabel("Action Movies Watched")
ax.set_ylabel("Comedy Movies Watched")
ax.set_title("Customer Movie Preferences with Principal Components (Color-Coded)")
ax.grid(True)

for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    ax.arrow(pca.mean_[0], pca.mean_[1], v[0], v[1],
              head_width=0.3, head_length=0.3, color='red')

st.pyplot(fig)

# Display PCA results
st.subheader("PCA Results")
st.write("### Principal Components:")
st.write(pca.components_)
st.write("### Explained Variance Ratio:")
st.write(pca.explained_variance_ratio_)
