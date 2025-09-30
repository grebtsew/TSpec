# Use official Python image
FROM python:3.12-slim

# Set work directory
WORKDIR /usr/src/app

# Copy requirements first to leverage Docker cache
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application
COPY . .

# Default command
CMD ["python", "TSpec.py" , "--address", "0.0.0.0"]
