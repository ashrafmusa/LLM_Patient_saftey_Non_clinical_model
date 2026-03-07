# Dockerfile for reproducible deployment
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY . .

# Expose ports for services
EXPOSE 8000 8501

# Default command: run tests
CMD ["python", "-m", "pytest", "tests/", "-v"]

# Optional commands:
# - Interactive evaluation: python -c "from src.evaluate import evaluate; evaluate(n=200)"
# - API server: uvicorn src.app:app --host 0.0.0.0 --port 8000
# - Streamlit UI: streamlit run src/ui.py --server.port 8501
