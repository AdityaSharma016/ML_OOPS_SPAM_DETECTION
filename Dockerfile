FROM python:3.12-slim

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY train_model.py .

COPY templates ./templates
COPY static ./static

# copy them here as well.

EXPOSE 10000

CMD ["python", "main.py"]