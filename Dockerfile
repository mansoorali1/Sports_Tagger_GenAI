FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY start.sh .

# Create empty artifacts directory — filled at runtime by download_artifacts.py
RUN mkdir -p artifacts

RUN chmod +x start.sh

EXPOSE 8000
EXPOSE 8501

CMD ["./start.sh"]
