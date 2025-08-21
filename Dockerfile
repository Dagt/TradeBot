FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src
# Include monitoring package so API can import monitoring.* modules
COPY monitoring/ ./monitoring
# Ensure both src and project root are on PYTHONPATH
ENV PYTHONPATH=/app/src:/app

CMD ["uvicorn", "tradingbot.apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
