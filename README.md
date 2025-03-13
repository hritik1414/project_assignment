# FastAPI-based ML Inference Service  

This project provides a FastAPI-based web service for running machine learning inferences using a pre-trained model. It also includes a Streamlit interface for user-friendly interaction.  

## Prerequisites  

Ensure you have the following installed on your system:  

- Python 3.9 or later  
- pip (Python package manager)  
- Virtual environment tool (optional but recommended)  
- Docker (if running with Docker)  

---

## Running the Project Without Docker  

### 1. Clone the Repository  

git clone <repository_url>
cd <repository_name>
2. Create and Activate a Virtual Environment (Recommended)
On macOS/Linux:

python3 -m venv venv
source venv/bin/activate
On Windows (Command Prompt):

python -m venv venv
venv\Scripts\activate
3. Install Dependencies
pip install --no-cache-dir -r requirements.txt
4. Running the FastAPI Server
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
By default, the API will be available at http://127.0.0.1:8000.
You can access the interactive API documentation at http://127.0.0.1:8000/docs.

5. Running the Streamlit Frontend
streamlit run streamlit.py
This will launch a web-based interface where users can interact with the model.

Troubleshooting
# If you encounter missing package errors (e.g., ModuleNotFoundError for `click` or `h11`), try reinstalling dependencies:  
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt
Running the Project with Docker

1. Clone the Repository
git clone <repository_url>
cd <repository_name>
2. Build and Run the Docker Container
# Build the Docker image  
docker build -t fastapi-ml-app .

# Run the container  
docker run -p 8000:8000 fastapi-ml-app
The API will now be accessible at http://127.0.0.1:8000.
The interactive API documentation will be available at http://127.0.0.1:8000/docs.

3. Running the Streamlit Frontend in Docker
docker run -p 8501:8501 fastapi-ml-app streamlit run streamlit.py
The Streamlit interface will be available at http://127.0.0.1:8501.

Notes

The Dockerfile is set up to use Python 3.9. If you need GPU support, consider using an official PyTorch image with CUDA.
Ensure all dependencies in requirements.txt are correctly installed to avoid missing module errors.
