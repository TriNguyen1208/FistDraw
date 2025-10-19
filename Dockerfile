from python:3.12-slim

workdir /app

copy src/ ./src

run apt-get update \
    && apt-get install -y python3 python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --no-cache-dir -r src/requirements.txt

cmd ["python3", "-m", "src.main"]