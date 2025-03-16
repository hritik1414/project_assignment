# MyModel - Trained Deep Learning Model

## 📌 Download the Model
You can download the trained model from Google Drive:

🔗 **([myModel.pth](https://drive.google.com/file/d/1ccyfQCrYv98a_o5rUgNqfyJwQnjQ6660/view?usp=sharing))**

## 📖 Usage Instructions
1. Download the `myModel.pth` file.
2. Place it in the appropriate directory.
3. Load the model in Python:
   
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

## Running the Project with Docker
If you prefer using Docker, here’s how to do it:

### 1. Clone the Repository

Download the project files and enter the folder:
```bash
git clone <repository_url>
cd <repository_name>
```
### 2. Build and Run the Docker Container

Build the Docker image and start the container:
```bash
# Build the Docker image
docker build -t <image-name> .

# Run the container
docker run -p 8000:8000 <image-name>
```
The API will be available at http://127.0.0.1:8000. You can see the interactive docs at http://127.0.0.1:8000/docs.

### 3. Run the Streamlit Frontend in Docker

To use the Streamlit interface with Docker, run this:
```bash
docker run -p 8501:8501 <image-name> streamlit run streamlit.py
```
# Postman API Testing Instructions

You can test the FastAPI backend using Postman by following these steps:

## 1) Open Postman and Create a New Request
Select POST request. <br>
Enter the request URL: <br>
http://127.0.0.1:8000/predict <br>
## 2) Set Up Authentication
Go to the Authorization tab. <br>
Select Basic Auth. <br>
Enter the following credentials: <br>
Username: admin <br>
Password: secret <br>
## 3) Upload an Image
Navigate to the Body tab. <br>
Select form-data. <br>
Add a new key with: <br>
Key: file <br>
Type: File <br>
Value: Select an image file (.png, .jpg, .jpeg). <br>
## 4) Send the Request
Click on the Send button. <br>
If successful, you will receive a JSON response with the predicted class: <br>
```bash
{
  "predicted_class": "<class_name>"
}
```
# Explainability

## Explainability using Grad-CAM (gradio_gradcam.py)

Uses Grad-CAM to visualize the model’s decision-making process.

Extracts the third-to-last feature layer as the target for Grad-CAM.

Generates a heatmap overlay highlighting important regions for classification.

Displays both the original and Grad-CAM visualized images side by side.

Run the following command to launch the Gradio interface for prediction and Grad-CAM visualization:
### For Mac:
```bash
python3 gradio_gradcam.py
```

### For Windows:
```bash
python gradio_gradcam.py
```

# Notes

The Dockerfile uses Python 3.9.

If you need GPU support, consider using a PyTorch image with CUDA.

Ensure all packages in requirements.txt are installed properly to avoid missing module errors.
