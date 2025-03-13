import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette.status import HTTP_401_UNAUTHORIZED
import shutil
import os
import uuid
import secrets
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Import your inference function from inference.py
from inference import inference

app = FastAPI()
security = HTTPBasic()

# Set up basic authentication credentials
USERNAME = "admin"
PASSWORD = "secret"

def get_current_username(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, USERNAME)
    correct_password = secrets.compare_digest(credentials.password, PASSWORD)
    if not (correct_username and correct_password):
        logger.warning("Authentication failed for username: %s", credentials.username)
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    logger.info("User authenticated: %s", credentials.username)
    return credentials.username

@app.post("/predict")
async def predict(file: UploadFile = File(...), username: str = Depends(get_current_username)):
    # Validate file extension
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        logger.error("Invalid file type: %s", file.filename)
        raise HTTPException(status_code=400, detail="Invalid file type. Only .png, .jpg, .jpeg are allowed")
    
    # Save the uploaded file to a temporary location
    temp_filename = f"temp_{uuid.uuid4().hex}_{file.filename}"
    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info("File saved temporarily as: %s", temp_filename)

        # Call your inference function on the saved file
        predicted_class = inference(temp_filename)
        logger.info("Inference successful: %s", predicted_class)
    except Exception as e:
        logger.exception("Error during inference: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
                logger.info("Temporary file removed: %s", temp_filename)
        except Exception as cleanup_error:
            logger.error("Error cleaning up temporary file: %s", cleanup_error)
    
    return {"predicted_class": predicted_class}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
