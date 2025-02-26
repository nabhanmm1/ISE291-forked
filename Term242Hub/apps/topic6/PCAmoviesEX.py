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
st.subheader("Interpreting Principal Component 1")
st.write("Observing the PCA-transformed data, we notice that PC1 appears to represent a customer's inclination towards either Action or Comedy movies.")
st.write("- Higher values of PC1 correspond to individuals who watch more Action movies than Comedy movies.")
st.write("- Lower values of PC1 correspond to individuals who watch more Comedy movies than Action movies.")
st.write("- A PC1 value close to 0 corresponds to individuals who watch an equal number of Action and Comedy movies.")
st.write("This suggests that PC1 can be interpreted as a 'Taste Preference Score,' where positive values indicate an inclination towards Action movies and negative values indicate an inclination towards Comedy movies.")

st.subheader("Interpreting Principal Component 2")
st.write("Examining PC2, we observe that it corresponds to the total number of movies watched by an individual, regardless of genre.")
st.write("- Higher values of PC2 indicate individuals who have watched more movies overall (both Action and Comedy).")
st.write("- Lower values of PC2 indicate individuals who have watched fewer movies overall.")
st.write("Since most individuals in the dataset have watched a similar total number of movies, the variance in PC2 is relatively low compared to PC1. This explains why PC2 captures much less variance.")

st.subheader("Why This Makes Sense")
st.write("PCA identifies the direction of maximum variance in the data. Since action and comedy movie counts are the two defining features in this dataset, PCA naturally finds the dominant trend, which happens to be how much an individual leans towards one genre over the other.")
st.write("The second principal component (PC2), which explains the remaining variance, would capture a different relationship, likely related to overall variability in watching behavior rather than preference.")

st.subheader("PCA Results & Explanation")
st.write("### Principal Components:")
st.write(pca.components_)
st.write("Each row in the matrix represents a principal component. These vectors indicate the new directions along which the data is aligned after transformation.")

st.write("### Explained Variance Ratio:")
st.write(pca.explained_variance_ratio_)
st.write("The explained variance ratio tells us how much of the total variance in the data is captured by each principal component. The sum of these values equals 1, meaning all variance in the original dataset is retained across the new dimensions.")

# Visualizing variance explained
st.subheader("Further Confirmation: Correlation Analysis")
st.write("To further validate our interpretation, we can examine the correlation between PC1 and the difference between Action and Comedy movies watched (Action - Comedy). If PC1 truly represents a 'Taste Preference Score,' we should observe a strong correlation.")

# Compute the difference between Action and Comedy
movie_diff = data[:, 0] - data[:, 1]

# Scatter plot to visualize correlation
fig, ax = plt.subplots()
ax.scatter(movie_diff, principal_components[:, 0], c=colors)
ax.set_xlabel("Action - Comedy Movies Watched")
ax.set_ylabel("PC1 Value")
ax.set_title("Correlation between Movie Preference Difference and PC1")
ax.grid(True)
st.pyplot(fig)

st.write("From the scatter plot, we can see that PC1 is highly correlated with the difference between Action and Comedy movies watched. This further supports our interpretation that PC1 represents the customer's taste preference.")

st.subheader("Further Confirmation: PC2 and Total Movies Watched")
st.write("To validate our interpretation of PC2, we can examine its correlation with the total number of movies watched (Action + Comedy). If PC2 represents overall movie-watching behavior, we should see a strong correlation.")

# Compute the total number of movies watched
total_movies = data[:, 0] + data[:, 1]

# Scatter plot to visualize correlation
fig, ax = plt.subplots()
ax.scatter(total_movies, principal_components[:, 1], c=colors)
ax.set_xlabel("Total Movies Watched")
ax.set_ylabel("PC2 Value")
ax.set_title("Correlation between Total Movies Watched and PC2")
ax.grid(True)
st.pyplot(fig)

st.write("From the scatter plot, we can see that PC2 is highly correlated with the total number of movies watched. This confirms our interpretation that PC2 represents overall movie-watching behavior.")

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
