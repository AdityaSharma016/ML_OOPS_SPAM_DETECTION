# 🚨 Spam Message Detector

An end-to-end Machine Learning and MLOps application that detects whether a text message is **Spam** or **Safe** using Natural Language Processing and Machine Learning.

The application provides a web-based interface where users can enter a message and receive a real-time prediction along with a model probability score.

The project demonstrates **FastAPI, Docker, automated testing, GitHub Actions CI/CD, and cloud deployment using Render**.

---

## 🌐 Live Demo

🚀 **Live Application:**  
https://ml-oops-spam-detection-1.onrender.com

📂 **GitHub Repository:**  
https://github.com/AdityaSharma016/ML_OOPS_SPAM_DETECTION

---

## 📸 Project Screenshots

### 🏠 Application Interface

![Spam Message Detector](screenshots/home.png)

### 🚨 Spam Detection

![Spam Prediction](screenshots/spam-prediction.png)

### ✅ Safe Message Detection

![Safe Prediction](screenshots/safe-prediction.png)

---

## 📌 Project Overview

Spam messages are a common problem in SMS, email, and online communication platforms.

This project uses **Natural Language Processing (NLP)** and **Machine Learning** to classify text messages into two categories:

- ✅ Safe Message
- 🚨 Spam Message

Users can enter a message through the web interface, and the application sends it to a FastAPI backend for prediction.

The backend processes the message using a trained **TF-IDF vectorizer** and **Multinomial Naive Bayes classifier**, then returns the prediction and model probability score.

The application is containerized using **Docker** and deployed through a CI/CD pipeline using **GitHub Actions and Render**.

---

## ✨ Features

- 🔍 Real-time spam message detection
- 🤖 Machine Learning-based text classification
- 🔤 TF-IDF text vectorization
- 🧠 Multinomial Naive Bayes classifier
- 📊 Model probability score
- ⚡ FastAPI REST API
- 🎨 Responsive web interface
- 🧪 Automated testing using Pytest
- 🐳 Docker containerization
- 🔄 GitHub Actions CI/CD pipeline
- ☁️ Cloud deployment using Render
- 📦 ML model serialization using Joblib
- 🔧 Automatic model training when model files are unavailable
- 🌐 FastAPI server configuration for cloud deployment

---

## 🚦 Project Status

- ✅ Machine Learning model implemented
- ✅ FastAPI REST API implemented
- ✅ Web interface implemented
- ✅ Automated tests implemented
- ✅ Docker containerization implemented
- ✅ GitHub Actions CI/CD implemented
- ✅ Render deployment implemented
- 🚧 Advanced MLOps features planned

---

# 🧠 Machine Learning

The project uses a **Multinomial Naive Bayes classifier** with **TF-IDF vectorization** for spam message classification.

## 🔤 TF-IDF Vectorization

TF-IDF stands for **Term Frequency-Inverse Document Frequency**.

It converts text messages into numerical feature vectors that can be processed by the Machine Learning model.

The current implementation uses:

```python
TfidfVectorizer(
    stop_words="english"
)
