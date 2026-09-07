FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY data ./data
COPY train_model.py .

RUN python train_model.py

COPY main.py .
COPY templates ./templates
COPY static ./static

EXPOSE 10000

CMD ["python", "main.py"]