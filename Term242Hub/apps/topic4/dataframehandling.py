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
    df = pd.read_csv(uploaded_file)
    st.write("### Full Dataset Preview:")
    st.dataframe(df)

    # Step 2: Data Slicing Interface
    st.subheader("Data Slicing")

    # Select columns to display
    selected_columns = st.multiselect("Select columns to view:", df.columns, default=df.columns.tolist())

    # User-friendly filtering interface
    st.write("### Apply Filters")
    
    filters = []
    num_filters = st.number_input("Number of conditions:", min_value=0, max_value=5, value=0, step=1)

    logic_operators = ["AND"] * (num_filters - 1)  # Default to AND between conditions

    for i in range(num_filters):
        col = st.selectbox(f"Select column {i+1}:", df.columns, key=f"col_{i}")
        condition = st.selectbox(f"Select condition for {col}:", ["=", "!=", ">", "<", ">=", "<="], key=f"cond_{i}")
        
        if condition == "=":
            unique_values = df[col].dropna().unique().tolist()
            value = st.multiselect(f"Select value(s) for {col}:", unique_values, key=f"val_{i}")
            value_str = " | ".join([f'`{col}` == "{v}"' for v in value]) if value else ""
        else:
            value = st.text_input(f"Enter value for {col}:", key=f"val_{i}")
            value_str = f'`{col}` {condition} "{value}"' if value else ""
        
        filters.append(value_str)

        if i < num_filters - 1:
            logic_operators[i] = st.selectbox(f"Select logical operator after condition {i+1}:", ["AND", "OR"], key=f"logic_{i}")
    
    filtered_df = df[selected_columns]  # Default is all selected columns
    
    if filters:
        non_empty_filters = [f for f in filters if f]
        query_string = f' {logic_operators[0]} '.join(non_empty_filters) if logic_operators and non_empty_filters else " ".join(non_empty_filters)
        
        try:
            if query_string:
                filtered_df = df.query(query_string)[selected_columns]
        except Exception as e:
            st.error(f"Invalid condition: {e}")

    st.write("### Sliced DataFrame Preview:")
    st.dataframe(filtered_df)

    # Step 3: Statistical Summaries
    st.subheader("Statistical Summaries")

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
    if "plot_count" not in st.session_state:
        st.session_state["plot_count"] = 1

    st.subheader("Plotting Dashboard")
    
    for i in range(st.session_state["plot_count"]):
        st.write(f"### Plot {i+1}")

        data_choice = st.radio(f"Choose dataset for Plot {i+1}:", ["Full Data", "Sliced Data"], key=f"data_choice_{i}")
        plot_data = df if data_choice == "Full Data" else filtered_df

        plot_type = st.selectbox(f"Select plot type for Plot {i+1}:", 
                                 ["Histogram", "Countplot", "Boxplot", "Scatterplot", "Lineplot"], 
                                 key=f"plot_type_{i}")

        if plot_type in ["Histogram", "Countplot", "Boxplot"]:
            x_col = st.selectbox(f"Select a categorical or numerical column for Plot {i+1}:", 
                                 plot_data.columns, key=f"x_col_{i}")
            hue_col = st.selectbox(f"Optional: Select hue column for Plot {i+1}:", 
                                   ["None"] + list(plot_data.columns), key=f"hue_col_{i}")

        elif plot_type in ["Scatterplot", "Lineplot"]:
            x_col = st.selectbox(f"Select X-axis column for Plot {i+1}:", 
                                 plot_data.columns, key=f"x_col_{i}")
            y_col = st.selectbox(f"Select Y-axis column for Plot {i+1}:", 
                                 plot_data.columns, key=f"y_col_{i}")
            hue_col = st.selectbox(f"Optional: Select hue column for Plot {i+1}:", 
                                   ["None"] + list(plot_data.columns), key=f"hue_col_{i}")

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

    if st.button("Add another plot"):
        st.session_state["plot_count"] += 1
        st.rerun()
