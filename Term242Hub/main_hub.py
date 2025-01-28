import streamlit as st
import os
import importlib.util

st.title("ISE 291 Term 242 Section F22 Streamlit Hub")
st.markdown("Welcome to the class's Streamlit Hub! Use the sidebar to navigate.")

# Sidebar Navigation
st.sidebar.title("Navigation")

# Dynamic Topics
topics = [f for f in os.listdir("apps") if os.path.isdir(os.path.join("apps", f))]
topic = st.sidebar.selectbox("Choose a Topic", topics)

# Sub-Apps for Selected Topic
sub_apps_path = os.path.join("apps", topic)
sub_apps = [f for f in os.listdir(sub_apps_path) if f.endswith(".py")]

sub_app = st.sidebar.selectbox("Choose a Sub-App", sub_apps)

# Load and Run the Selected Sub-App
app_path = os.path.join(sub_apps_path, sub_app)
spec = importlib.util.spec_from_file_location("sub_app", app_path)
sub_app_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sub_app_module)
