FROM ubuntu:24.04

# Install all
COPY ./souptSDR/install_full.sh /install_full.sh
RUN chmod +x /install_full.sh
RUN /install_full.sh

# Install python packages
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application
COPY . .

# Default command
CMD ["python", "./src/TSpec.py" , "--address", "0.0.0.0"]
