# Dockerfile — training execution
FROM python:3.10-slim
WORKDIR /app

# dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# code
COPY . .

CMD ["python", "src/train.py"]
