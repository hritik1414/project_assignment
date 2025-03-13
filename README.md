FastAPI-based ML Inference Service

This project provides a FastAPI-based web service for running machine learning inferences using a pre-trained model. It includes a Streamlit interface for user-friendly interaction.

Prerequisites

Ensure you have the following installed on your system:

Python 3.9 or later
pip (Python package manager)
Virtual environment tool (optional but recommended)
Setup Instructions

Clone the Repository
git clone <repository_url>
cd <repository_name>
Create and Activate a Virtual Environment (Recommended)
On macOS/Linux:
python3 -m venv venv
source venv/bin/activate
On Windows (Command Prompt):
python -m venv venv
venv\Scripts\activate
Install Dependencies
pip install --no-cache-dir -r requirements.txt
Running the FastAPI Server

Start the FastAPI backend using Uvicorn:

uvicorn app:app --host 0.0.0.0 --port 8000 --reload
By default, the API will be available at http://127.0.0.1:8000.
You can access the interactive API documentation at http://127.0.0.1:8000/docs.

Running the Streamlit Frontend

To start the Streamlit interface, run:

streamlit run streamlit.py
This will launch a web-based interface where users can interact with the model.

Troubleshooting

If you encounter missing package errors (e.g., ModuleNotFoundError for click or h11), try reinstalling dependencies:
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt
If issues persist, ensure you’re inside the virtual environment and that all dependencies are correctly installed.
