# Use a slim Python base image.
FROM python:3.12-slim-bookworm

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
# Using --no-cache-dir to save space
# Using --upgrade pip to ensure pip is up-to-date
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy your application code
# Ensure predict.py and heart.csv (if needed) are in the same directory
COPY predict.py .
# If your predict.py *directly* reads heart.csv (it shouldn't for inference,
# but just in case), you'd need to copy it:
# COPY heart.csv .

# Expose the port that Uvicorn will run on
EXPOSE 8000

# Set environment variable for MLflow Tracking URI
# Replace with your MLflow Tracking Server's actual URL if it's not localhost
ENV MLFLOW_TRACKING_URI="http://172.17.0.1:5000"
# Note: 'host.docker.internal' is a special DNS name for Docker Desktop users
# to access the host machine. If on Linux, use your host's IP address (e.g., 172.17.0.1)
# or bridge network configurations. For local development, this is common.

# Command to run the application
# `--host 0.0.0.0` is essential for Docker containers to be accessible externally
CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000"]