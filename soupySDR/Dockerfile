# --- Basimage ---
    FROM python:3.11-slim

    # --- Installera systemberoenden ---
    RUN apt-get update && \
        apt-get install -y rtl-sdr libusb-1.0-0 git && \
        apt-get clean && rm -rf /var/lib/apt/lists/*
    
    # --- Installera Pythonberoenden ---
    COPY ./requirements.txt /requirements.txt
    RUN pip install --no-cache-dir -r /requirements.txt
    RUN pip install soupysdr
    
    WORKDIR /app
    
    COPY . .
    

    CMD ["python3", "./src/TSpec.py"]
    