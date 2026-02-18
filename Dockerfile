FROM python:3.12-slim

WORKDIR /app

# Install required system libraries for LightGBM
RUN apt-get update && \
    apt-get install -y gcc g++ libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --default-timeout=300 --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
