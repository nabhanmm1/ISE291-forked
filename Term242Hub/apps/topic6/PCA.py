import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA

# Ensure Matplotlib uses a non-interactive backend for Streamlit
matplotlib.use("Agg")

def main():
    st.title("Principal Component Analysis (PCA) of Customer Movie Preferences")

    st.markdown("""
    This app performs Principal Component Analysis (PCA) on a dataset representing customer movie preferences. 
    It aims to visualize and understand how customers are clustered based on their viewing habits of Action and Comedy movies.
    
    **Dataset:**
    - Each data point represents a customer.
    - Two features are analyzed: "Action Movies Watched" and "Comedy Movies Watched".
    - The goal is to reduce the dimensionality of this data and visualize the main trends.
    
    **How to Interpret the Plots:**
    - **Original Data:** Shows the raw data points with each color representing a different customer.
    - **PCA Transformed Data:** Shows the data projected onto the principal components (PC1 and PC2).
    - **PCA Vectors on Original Data:** Shows the original data with the principal component vectors overlaid, indicating the directions of maximum variance.
    
    **PCA Results:**
    - Displays the principal component vectors and the explained variance ratio, which indicates how much of the original variance is captured by each principal component.
    """)

    # Data
    data = np.array([
        [10, 2],
        [7, 5],
        [2, 8],
        [1, 10],
        [8, 3],
        [5, 5]
    ])

    # Assign colors to each data point
    colors = ['red', 'green', 'blue', 'purple', 'orange', 'black']

    # Debugging output
    st.write("Dataset Shape:", data.shape)
    st.write("Original Data:", data)

    # Original Data Plot
    st.subheader("Original Data: Action Movies vs. Comedy Movies")
    st.markdown("This plot shows the raw data points, where each color represents a different customer.")

    fig1, ax1 = plt.subplots()
    ax1.scatter(data[:, 0], data[:, 1], c=colors)
    ax1.set_xlabel("Action Movies Watched")
    ax1.set_ylabel("Comedy Movies Watched")
    ax1.set_title("Color-Coded Customer Movie Preferences")
    ax1.grid(True)
    
    st.pyplot(fig1)
    plt.close(fig1)  # Ensure figure is closed after displaying

    # PCA Calculation
    pca = PCA(n_components=2)
    pca.fit(data)
    principal_components = pca.transform(data)

    # Debugging output
    st.write("PCA Components:", pca.components_)
    st.write("PCA Explained Variance Ratio:", pca.explained_variance_ratio_)

    # PCA Transformed Data Plot
    st.subheader("PCA Transformed Data: PC1 vs. PC2")
    st.markdown("This plot shows the data transformed into the principal component space. PC1 and PC2 represent the directions of maximum variance.")

    fig2, ax2 = plt.subplots()
    ax2.scatter(principal_components[:, 0], principal_components[:, 1], c=colors)
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.set_title("PCA of Customer Movie Preferences (Color-Coded)")
    ax2.grid(True)
    
    st.pyplot(fig2)
    plt.close(fig2)

    # PCA with Vectors on Original Data
    st.subheader("PCA Vectors on Original Data")
    st.markdown("This plot shows the original data with the principal component vectors overlaid. The vectors indicate the directions of maximum variance.")

    fig3, ax3 = plt.subplots()
    ax3.scatter(data[:, 0], data[:, 1], c=colors)
    ax3.set_xlabel("Action Movies Watched")
    ax3.set_ylabel("Comedy Movies Watched")
    ax3.set_title("Customer Movie Preferences with Principal Components (Color-Coded)")
    ax3.grid(True)

    # Plot PCA vectors (principal components)
    for length, vector in zip(pca.explained_variance_ratio_, pca.components_):
        v = vector * np.sqrt(length) * 3  # Scaling adjustment
        ax3.arrow(pca.mean_[0], pca.mean_[1], v[0], v[1],
                  head_width=0.2, head_length=0.2, color='red')

    st.pyplot(fig3)
    plt.close(fig3)

    # Display PCA Results
    st.subheader("PCA Results")
    st.write("Principal Components:")
    st.write(pca.components_)
    st.write("Explained Variance Ratio:")
    st.write(pca.explained_variance_ratio_)

if __name__ == "__main__":
    main()
