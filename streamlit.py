# streamlit_app.py
import streamlit as st
import requests

st.title("Image Classification with FastAPI")
st.write("Upload an image to get a prediction from the deployed model.")

# Image upload widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Input fields for Basic Auth credentials
username = st.text_input("Username", value="admin")
password = st.text_input("Password", type="password", value="secret")

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    if st.button("Predict"):
        # Prepare file for upload
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        # Make POST request to FastAPI endpoint (adjust URL if needed)
        response = requests.post("http://localhost:8000/predict", files=files, auth=(username, password))
        if response.status_code == 200:
            result = response.json()
            st.write("Predicted Class:", result.get("predicted_class"))
        else:
            st.error("Error: " + response.text)
