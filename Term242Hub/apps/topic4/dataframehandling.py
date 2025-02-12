import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set up the Streamlit app
st.title("DataFrame Handling & Visualization App")
st.write("Upload a dataset to explore, slice, and visualize your data.")

# Step 1: File Upload
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

if uploaded_file:
    # Load the dataset
    df = pd.read_csv(uploaded_file)
    st.write("### Full Dataset Preview:")
    st.dataframe(df)

    # Step 2: Data Slicing Interface
    st.subheader("Data Slicing")

    # Column selection
    selected_columns = st.multiselect("Select columns to view:", df.columns, default=df.columns.tolist())

    # Conditional filtering
    st.write("### Conditional Filtering (Optional)")
    condition_str = st.text_area("Enter condition (Python syntax)", value="")  
    filtered_df = df[selected_columns]  # Default is all selected columns

    if condition_str:
        try:
            filtered_df = df.query(condition_str)[selected_columns]
        except Exception as e:
            st.error(f"Invalid condition: {e}")

    st.write("### Sliced DataFrame Preview:")
    st.dataframe(filtered_df)

    # Step 3: Statistical Summaries
    st.subheader("Statistical Summaries")

    # Identify categorical and numerical columns
    categorical_cols = filtered_df.select_dtypes(include=["object"]).columns
    numerical_cols = filtered_df.select_dtypes(include=["number"]).columns

    if len(numerical_cols) > 0:
        st.write("### Numerical Column Summary:")
        st.write(filtered_df[numerical_cols].describe())

    if len(categorical_cols) > 0:
        st.write("### Categorical Column Summary:")
        for col in categorical_cols:
            st.write(f"**{col} Value Counts:**")
            st.write(filtered_df[col].value_counts())

    # Step 4: Interactive Plotting Dashboard
    st.subheader("Plotting Dashboard")
    plots = []
    add_plot = True

    while add_plot:
        st.write("### Add a New Plot")

        # Select dataset to plot (Full Data or Sliced Data)
        data_choice = st.radio("Choose dataset for plotting:", ["Full Data", "Sliced Data"], index=1)
        plot_data = df if data_choice == "Full Data" else filtered_df

        # Select plot type
        plot_type = st.selectbox("Select plot type:", ["Histogram", "Countplot", "Boxplot", "Scatterplot", "Lineplot"])

        # Define plot parameters based on type
        if plot_type in ["Histogram", "Countplot", "Boxplot"]:
            x_col = st.selectbox("Select a categorical or numerical column:", plot_data.columns)
            hue_col = st.selectbox("Optional: Select a column for hue (categorization):", ["None"] + list(plot_data.columns))

        elif plot_type in ["Scatterplot", "Lineplot"]:
            x_col = st.selectbox("Select X-axis column:", plot_data.columns)
            y_col = st.selectbox("Select Y-axis column:", plot_data.columns)
            hue_col = st.selectbox("Optional: Select a column for hue (categorization):", ["None"] + list(plot_data.columns))

        # Generate plot
        fig, ax = plt.subplots()
        if plot_type == "Histogram":
            sns.histplot(data=plot_data, x=x_col, hue=hue_col if hue_col != "None" else None, ax=ax, bins=20)
        elif plot_type == "Countplot":
            sns.countplot(data=plot_data, x=x_col, hue=hue_col if hue_col != "None" else None, ax=ax)
        elif plot_type == "Boxplot":
            sns.boxplot(data=plot_data, x=x_col, y=hue_col if hue_col != "None" else None, ax=ax)
        elif plot_type == "Scatterplot":
            sns.scatterplot(data=plot_data, x=x_col, y=y_col, hue=hue_col if hue_col != "None" else None, ax=ax)
        elif plot_type == "Lineplot":
            sns.lineplot(data=plot_data, x=x_col, y=y_col, hue=hue_col if hue_col != "None" else None, ax=ax)

        st.pyplot(fig)

        # Ask user if they want to add another plot
        add_plot = st.checkbox("Add another plot?", value=False)
