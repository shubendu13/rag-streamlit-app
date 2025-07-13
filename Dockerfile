# Use official Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# System-level dependencies
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy only requirements first (so pip install layer is cached)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Now copy the rest of the code
COPY . .

# Run prepare_index.py once at container start, then launch Streamlit
CMD ["bash", "-c", "python prepare_index.py && streamlit run app.py --server.port=8501 --server.address=0.0.0.0"]
