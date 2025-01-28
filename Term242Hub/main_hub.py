import streamlit as st

st.title("ISE 291 Term 242 Section F22 Streamlit Hub")
st.markdown("Welcome to your group’s Streamlit Hub! Use the sidebar to navigate.")

# Sidebar Navigation
st.sidebar.title("Navigation")
section = st.sidebar.selectbox("Choose a Topic", [
    "Welcome",
    "Topic 1 - Exploratory Data Analysis",
    "Topic 2 - Machine Learning",
])

# Display Section Content
if section == "Welcome":
    st.subheader("Welcome to the Hub!")
    st.write("Instructions on how to use the hub.")
elif section == "Topic 1 - Exploratory Data Analysis":
    st.subheader("Topic 1")
    st.write("Navigate to apps for Topic 1.")
elif section == "Topic 2 - Machine Learning":
    st.subheader("Topic 2")
    st.write("Navigate to apps for Topic 2.")
