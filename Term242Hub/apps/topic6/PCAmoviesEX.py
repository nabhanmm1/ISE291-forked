import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Streamlit App Setup
st.title("PCA Visualization App")
st.write("This interactive application demonstrates Principal Component Analysis (PCA) applied to customer movie preferences. PCA is used to reduce the dimensionality of data while preserving the most critical information.")

# Data
st.subheader("Original Data")
st.write("The dataset consists of customer movie preferences in two genres: Action and Comedy. Each point represents a customer, with the x-axis representing the number of Action movies watched and the y-axis representing the number of Comedy movies watched.")

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
ax.set_title("Customer Movie Preferences")
ax.grid(True)
st.pyplot(fig)

# Compute PCA
st.subheader("PCA Computation")
st.write("PCA finds a new set of axes (principal components) that best explain the variance in the data.")
pca = PCA(n_components=2)
pca.fit(data)
principal_components = pca.transform(data)

# Scatter plot of PCA-transformed data
st.subheader("Transformed Data (PCA Representation)")
st.write("After applying PCA, the data is rotated to align with the directions of maximum variance. The axes now represent the principal components.")
fig, ax = plt.subplots()
ax.scatter(principal_components[:, 0], principal_components[:, 1], c=colors)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("PCA of Customer Movie Preferences")
ax.grid(True)
st.pyplot(fig)

# Visualizing principal component vectors on the original data
st.subheader("Principal Components on Original Data")
st.write("The arrows represent the directions of the principal components, showing how PCA reorients the data along axes of maximum variance.")
fig, ax = plt.subplots()
ax.scatter(data[:, 0], data[:, 1], c=colors)
ax.set_xlabel("Action Movies Watched")
ax.set_ylabel("Comedy Movies Watched")
ax.set_title("Customer Movie Preferences with Principal Components")
ax.grid(True)

for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    ax.arrow(pca.mean_[0], pca.mean_[1], v[0], v[1],
              head_width=0.3, head_length=0.3, color='red')

st.pyplot(fig)

# Display PCA results with explanations
st.subheader("PCA Results & Explanation")
st.write("### Principal Components:")
st.write(pca.components_)
st.write("Each row in the matrix represents a principal component. These vectors indicate the new directions along which the data is aligned after transformation.")

st.write("### Explained Variance Ratio:")
st.write(pca.explained_variance_ratio_)
st.write("The explained variance ratio tells us how much of the total variance in the data is captured by each principal component. The sum of these values equals 1, meaning all variance in the original dataset is retained across the new dimensions.")

# Visualizing variance explained
st.subheader("Variance Explained by Each Principal Component")
st.write("The following bar chart represents the proportion of total variance explained by each principal component.")
fig, ax = plt.subplots()
ax.bar(["PC1", "PC2"], pca.explained_variance_ratio_, color=['blue', 'green'])
ax.set_ylabel("Proportion of Variance Explained")
ax.set_title("Variance Explained by Principal Components")
st.pyplot(fig)

st.write("### Understanding Variance Preservation:")
st.write("The total variance in the original dataset is equal to the sum of the variances in the principal components. This means that while PCA transforms the data, it retains all the variance present in the original dimensions.")

st.write("Mathematically, this is given by:")
st.latex(r"\sum Var_{original} = \sum Var_{PCs}")
st.write("This confirms that PCA is a lossless transformation in terms of variance distribution across dimensions.")
