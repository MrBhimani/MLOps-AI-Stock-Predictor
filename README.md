---
title: AI Stock Prediction System using MLOPs
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
---

This project uses **pre-commit hooks** to enforce code quality standards.
Enabled tools: black (code formatting), isort (import sorting), flake8 (linting), basic Git hygiene checks

### 🌱 Inspiration & Acknowledgement

This project was inspired by my experience **teaching Data Visualization** and Dashboard Design to students, where they successfully built insightful stock market dashboards using Tableau and Power BI. Observing their progress motivated me to extend this work further—from descriptive and visual analytics toward predictive intelligence using AI and MLOps practices.

As a learner, Nuqta represents my journey into AI-driven stock prediction, real-world ML system design, and end-to-end MLOps pipelines. The goal of this project is not only prediction accuracy, but also learning how production-ready ML systems are built, deployed, monitored, and improved continuously.

I am sincerely thankful to **Vladislav Naumov** for his guidance, mentorship, and continuous support. His teaching approach and encouragement toward innovation played a key role in motivating me to explore this domain and build something meaningful beyond the classroom.

## 📊 Data Management (DVC)

This project uses **DVC (Data Version Control)** for dataset management.

⚠️ **Raw datasets are NOT stored on GitHub**  
Only DVC metadata files (e.g. `data/raw.dvc`) are tracked in Git.

### Dataset availability
Market data (Open, High, Low, Close, Volume) is sourced in real time from Alpha Vantage via its public REST API. Users must obtain a free API key from Alpha Vantage and configure it locally to enable data ingestion.

# Nuqta | AI Market Insight & Stock Predictor

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/nvvy/nuqta-Stock-predictor)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-2CA5E0?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)

**Nuqta** is an advanced **End-to-End Machine Learning System** designed for real-time stock price prediction and market regime analysis. It leverages ensemble modeling (Linear Regression, Random Forest, SVM) and unsupervised learning to provide actionable financial insights through a premium, "Modern FinTech" aesthetic dashboard.

---

## 🚀 Live Demo

Check out the deployed application on Hugging Face Spaces:

👉 **[Nuqta Stock Predictor (Live App)](https://huggingface.co/spaces/nvvy/nuqta-Stock-predictor)**

---

## 🌟 Key Features

*   **📈 Multi-Model AI Predictions:**  
    combines **Regression** (price targets), **Classification** (trend direction), and **Clustering** (market volatility regimes) for robust decision support.
*   **⏱️ Real-Time Market Data:**  
    Fetches live stock data for global markets (USA, Pakistan, India, UK, etc.) using the **Alpha Vantage API**.
*   **🎨 Premium UI/UX:**  
    A highly responsive, glassmorphism-inspired interface built with **Streamlit** and custom CSS, featuring interactive **Plotly** charts.
*   **🔔 Smart Notifications:**  
    Integrated **Discord Alerts** to notify users of significant price movements and prediction updates.
*   **🔄 Automated MLOps Pipeline:**  
    Fully automated training and deployment pipelines using **Prefect** for orchestration and **GitHub Actions** for CI/CD.
*   **☁️ Cloud Native:**  
    Containerized with **Docker** and deployed seamlessly on cloud platforms.

---

## 🛠️ Tech Stack

*   **Frontend:** Streamlit, Plotly, HTML/CSS (Custom Styling)
*   **Backend & Logic:** Python, Scikit-Learn, Pandas, NumPy, SciPy
*   **Data Source:** Alpha Vantage, Yahoo Finance (yfinance)
*   **DevOps & MLOps:** Docker, GitHub Actions, Prefect, Hugging Face Hub

---

## � Project Structure

```bash
MLOps-AI-Stock-Predictor/
├── .github/              # CI/CD workflows (GitHub Actions)
├── data/                 # Raw and processed datasets
├── models/               # Serialized trained models (.pkl) generate after running
├── src/                  # Source code modules
│   ├── api/              # API endpoints (if applicable)
│   ├── ingestion/        # Data fetching scripts
│   ├── processing/       # Feature engineering & preprocessing
│   ├── models/           # Model definitions & training logic
│   ├── orchestration/    # Prefect flows & monitoring scripts
├── tests/                # Unit and integration tests
├── app.py                # Main Streamlit application entry point
├── Dockerfile            # Container configuration
├── requirements.txt      # Python dependencies
├── docs/                 # Project Report
└── README.md             # Project documentation
```

---

## ⚙️ Installation & Setup

Follow these steps to run the project locally.

### 1. Prerequisites

*   Python 3.10 or higher
*   Git
*   [Alpha Vantage API Key](https://www.alphavantage.co/) (Free key available)

### 2. Clone the Repository

```bash
git clone https://github.com/MrBhimani/MLOps-AI-Stock-Predictor.git
cd MLOps-AI-Stock-Predictor
```

### 3. Set Up Environment

Create a `.env` file in the root directory and add your API keys:

```bash
# .env
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
HF_TOKEN=your_huggingface_token  # Optional: for cloud training
WEBHOOK_URL=your_discord_webhook   # Optional: for notifications
```

### 4. Install Dependencies

It is recommended to use a virtual environment:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 5. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 🐳 Docker Setup

You can also run the application using Docker to ensure a consistent environment.

```bash
# Build the image
docker build -t nuqta-predictor .

# Run the container
docker run -p 7860:7860 --env-file .env nuqta-predictor
```

Access the app at `http://localhost:7860`.

## Model Training

Model training is implemented using **PyTorch Lightning**

## Configuration Management (Hydra)

This project uses **Hydra** for configuration management.

All training parameters and data paths are defined in YAML configs.

## Experiment Tracking & Logging

This project uses **MLflow** for experiment tracking.

## Inference & Model Packaging

Trained models are saved as artifacts and reused for inference.

---
Thank you.
