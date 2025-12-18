FROM python:3.9-slim

WORKDIR /opt/ml/code

COPY src/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ .

ENV PYTHONUNBUFFERED=TRUE
