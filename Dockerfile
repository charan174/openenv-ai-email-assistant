FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy environment files
COPY . .

# Run inference script by default to show baseline
CMD ["python", "inference.py"]
