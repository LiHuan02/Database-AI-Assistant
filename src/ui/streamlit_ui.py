"""
Streamlit UI components placeholder: library selector, API key, model selector, file upload
"""

import streamlit as st

st.sidebar.header("Settings")
api_key = st.sidebar.text_input("API Key", type="password")
model = st.sidebar.selectbox("Model", ["gpt-4", "gpt-3.5-turbo"])

st.sidebar.markdown("---")

st.header("Libraries")
st.write("Library list and create/delete controls go here.")

st.header("Chat")
st.write("Chat area and conversation controls go here.")
