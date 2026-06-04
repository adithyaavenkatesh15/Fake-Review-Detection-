FROM python:3.9-slim

WORKDIR /app

# Ensure rust/build-essential for some pip packages if needed
RUN apt-get update && apt-get install -y build-essential curl

# Install dependencies
COPY pyproject.toml .
RUN pip install -e .

# Copy project files
COPY . .

# Expose FastAPI
EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
