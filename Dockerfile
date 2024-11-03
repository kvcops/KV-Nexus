FROM python:3.11-slim

WORKDIR /app

# Optional: Force pip to use Python 3.11
ENV PYTHON_VERSION=3.11
ENV PATH="/usr/local/python/3.11/bin:${PATH}"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]
