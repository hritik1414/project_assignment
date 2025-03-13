# FastAPI-based ML Inference Service

This project is a web service built with FastAPI that runs machine learning predictions using a pre-trained model. It also comes with a Streamlit interface to make it easy for users to interact with the model.

## Prerequisites

Before you start, make sure you have these installed on your computer:

- Python 3.9 or a newer version  
- `pip` (the Python package manager)  
- A virtual environment tool (this is optional but a good idea)  
- Docker (only if you want to use Docker)  

## Running the Project Without Docker

Follow these steps to set up and run the project without Docker:

### 1. Clone the Repository

Get the project files from the repository and go into the project folder:

```bash
git clone <repository_url>
cd <repository_name>
```

### 2. Create and Activate a Virtual Environment (Recommended)

Set up a virtual environment to keep things organized. Here’s how to do it:

On macOS or Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```
On Windows (using Command Prompt):
```bash
python -m venv venv
venv\Scripts\activate
```
### 3. Install Dependencies

Install all the required Python packages:
```bash
pip install --no-cache-dir -r requirements.txt
```
### 4. Run the FastAPI Server

Start the FastAPI server with this command:
``` bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
Once it’s running, you can visit the API at http://127.0.0.1:8000. For more details, check the interactive docs at http://127.0.0.1:8000/docs.

### 5. Run the Streamlit Frontend

Launch the Streamlit interface with this command:
```bash
streamlit run streamlit.py
```
This opens a web page where you can use the model easily.

### Troubleshooting

If you see errors like ModuleNotFoundError for packages (e.g., click or h11), try updating pip and reinstalling the packages:

```bash
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt
```
