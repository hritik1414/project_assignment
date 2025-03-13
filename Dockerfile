# Use an official Python image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy all project files into the container
COPY . .

# Install system dependencies (for OpenCV or similar packages)
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install dependencies without cache
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch separately (CPU version)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Expose port for FastAPI
EXPOSE 8000

# Run FastAPI using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
