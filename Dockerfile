FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY start.sh .

# Create runtime artifacts directory
RUN mkdir -p artifacts

# Make startup script executable
RUN chmod +x start.sh

# Streamlit environment variables
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Expose ports
EXPOSE 8501
EXPOSE 8000

# Hugging Face health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Start app
CMD ["./start.sh"]
