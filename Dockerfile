FROM python:3.11-slim

WORKDIR /app

# Install lightweight cloud dependencies
COPY requirements-cloud.txt .
RUN pip install --no-cache-dir -r requirements-cloud.txt

# Copy project files
COPY . .

# Expose port (default 8000)
EXPOSE 8000

ENV PORT=8000

# Start FastAPI and ChatGPT UI
CMD ["sh", "-c", "uvicorn server.app:app --host 0.0.0.0 --port ${PORT}"]
