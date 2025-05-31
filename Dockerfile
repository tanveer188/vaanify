# Use official Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /

# Install build dependencies for packages like webrtcvad
RUN apt-get update && apt-get install -y gcc build-essential python3-dev

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY . .

# Expose port (FastAPI default is 8000)
EXPOSE 8010

# Start server using Uvicorn
CMD ["python", "ai_voice_assitance.py"]