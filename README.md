<h1 align="center">🎌 Anime Recommender System 🎌</h1>
<p align="center">
  <img src="https://img.shields.io/badge/Build-Passing-brightgreen?style=for-the-badge&logo=github" />
  <img src="https://img.shields.io/badge/Made%20with-Python-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/DVC-Version%20Control-purple?style=for-the-badge&logo=dvc" />
  <img src="https://img.shields.io/badge/Cloud-GCP-F9AB00?style=for-the-badge&logo=google-cloud" />
  <img src="https://img.shields.io/badge/CI/CD-Jenkins-red?style=for-the-badge&logo=jenkins" />
  <img src="https://img.shields.io/badge/Containerized-Docker-2496ED?style=for-the-badge&logo=docker" />
</p>

---

## 🌟 Overview

This is a web-based **Anime Recommender System** built with machine learning techniques and deployed using a full MLOps pipeline:
- Jenkins CI/CD
- Docker
- DVC for versioning
- Google Cloud Kubernetes (GKE)
- Kubernetes YAML Deployment
- Flask Web UI

---

## 🧠 ML Overview

We used:
- Collaborative filtering (User-User & Anime-Anime encoding)
- Train/test splits
- Pickled model pipelines
- KNN-based recommendation engine

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| 🐍 Python | Core ML development |
| 🧪 Jenkins | Automation of CI/CD |
| 🐳 Docker | Containerization |
| ☁️ Google Cloud | Hosting via GKE |
| ☸️ Kubernetes | App deployment |
| 🧬 DVC | Data & model version control |
| 📈 Pandas, Numpy, Sklearn | ML stack |
| 📋 Flask | Web interface |
| 🔧 VS Code | Dev environment |

---
<details>
<summary><strong>📁 Project Structure</strong> (click to expand)</summary>
Anime-Recommender/ │ ├── 📄 application.py # Main Flask application ├── 📄 Dockerfile # Docker config for the app ├── 📄 Jenkinsfile # Jenkins pipeline definition ├── 📄 deployment.yaml # Kubernetes deployment file ├── 📄 requirements.txt # Project dependencies ├── 📄 setup.py # Setup for pip install ├── 📄 tester.py # Testing file ├── 📄 .gitignore ├── 📄 .dvcignore │ ├── 📁 dvc/ # DVC cache & tmp │ ├── 📁 cache/ │ ├── 📁 tmp/ │ ├── 📄 .gitignore │ └── 📁 config/ │ ├── 📁 artifacts/ # Stored ML artifacts │ ├── 📁 model/ │ ├── 📁 model_checkpoint/ │ ├── 📁 processed/ │ │ ├── anime_df.csv │ │ ├── rating_df.csv │ │ ├── synopsis_df.csv │ │ ├── anime2anime_encoded.pkl │ │ ├── anime2anime_decoded.pkl │ │ ├── user2user_encoded.pkl │ │ ├── user2user_decoded.pkl │ │ ├── X_train_array.pkl │ │ ├── X_test_array.pkl │ │ ├── y_train.pkl │ │ └── y_test.pkl │ ├── 📁 raw/ │ │ ├── anime.csv │ │ ├── animelist.csv │ │ └── anime_with_synopsis.csv │ └── 📁 weights/ │ ├── 📄 model_checkpoint.dvc ├── 📄 model.dvc ├── 📄 processed.dvc ├── 📄 raw.dvc ├── 📄 weights.dvc │ ├── 📁 config/ # Configuration management │ ├── pycache/ │ ├── init.py │ ├── config.yaml │ └── paths_config.py │ ├── 📁 custom-jenkins/ # Custom Jenkins with Docker │ └── 🐳 Dockerfile │ ├── 📁 logs/ # Log files │ ├── 📁 ml_project_02.egg-info/ # Package info (auto-generated) │ ├── 📁 notebook/ # Jupyter notebook for EDA │ ├── anime.ipynb │ └── weights.weights.h5 │ ├── 📁 pipeline/ # Pipelines for ML │ ├── pycache/ │ ├── init.py │ ├── prediction_pipeline.py │ └── training_pipeline.py │ ├── 📁 src/ # Source code │ ├── pycache/ │ ├── init.py │ ├── base_model.py │ ├── custom_exception.py │ ├── data_ingestion.py │ ├── data_processing.py │ ├── logger.py │ └── model_training.py │ ├── 📁 static/ # Static files (CSS) │ └── style.css │ ├── 📁 template/ # HTML templates │ └── index.html │ ├── 📁 utils/ # Helper utilities │ ├── pycache/ │ ├── init.py │ ├── common_functions.py │ └── helpers.py │ └── 📁 venv/ # Virtual environment

bash
Copy
Edit


</details>


---

## ⚙️ Setup Instructions

### ✅ Prerequisite
- Docker Desktop running
- VS Code or any IDE
- GCP CLI installed & configured

---

## 🛠️ Step 1: Jenkins with Docker-in-Docker (DinD)

Create a folder `custom_jenkins` and add this `Dockerfile`:

```Dockerfile
FROM jenkins/jenkins:lts

USER root

RUN apt-get update -y && \
    apt-get install -y apt-transport-https ca-certificates curl gnupg software-properties-common && \
    curl -fsSL https://download.docker.com/linux/debian/gpg | apt-key add - && \
    echo "deb [arch=amd64] https://download.docker.com/linux/debian bullseye stable" > /etc/apt/sources.list.d/docker.list && \
    apt-get update -y && \
    apt-get install -y docker-ce docker-ce-cli containerd.io && \
    apt-get clean

RUN groupadd -f docker && usermod -aG docker jenkins

RUN mkdir -p /var/lib/docker
VOLUME /var/lib/docker

USER jenkins

🔧 Build and Run Jenkins

cd custom_jenkins
docker build -t jenkins-dind .
docker run -d --name jenkins-dind ^
--privileged ^
-p 8080:8080 -p 50000:50000 ^
-v //var/run/docker.sock:/var/run/docker.sock ^
-v jenkins_home:/var/jenkins_home ^
jenkins-dind

🔑 Get Jenkins Password

docker logs jenkins-dind

Visit localhost:8080, paste the password, and install suggested plugins.

🧪 Step 2: Enable Python in Jenkins Container

docker exec -u root -it jenkins-dind bash

apt update -y
apt install -y python3 python3-pip python3-venv
ln -s /usr/bin/python3 /usr/bin/python
exit

docker restart jenkins-dind

☁️ Step 3: Install GCP SDK & Kubernetes CLI

docker exec -u root -it jenkins-dind bash

apt-get update
apt-get install -y curl apt-transport-https ca-certificates gnupg
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
echo "deb https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
apt-get update && apt-get install -y google-cloud-sdk kubectl google-cloud-sdk-gke-gcloud-auth-plugin

kubectl version --client
gcloud --version
exit

🛡️ Step 4: Docker Permissions

docker exec -u root -it jenkins-dind bash
groupadd docker
usermod -aG docker jenkins
usermod -aG root jenkins
exit
docker restart jenkins-dind


🐍 Step 5: App Dockerfile

FROM python:3.8-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    build-essential libatlas-base-dev libhdf5-dev libprotobuf-dev protobuf-compiler python3-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -e .

RUN python pipeline/training_pipeline.py

EXPOSE 5000
CMD ["python", "application.py"]

☸️ Step 6: Kubernetes Deployment YAML
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-app
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ml-app
  template:
    metadata:
      labels:
        app: ml-app
    spec:
      containers:
      - name: ml-app-container
        image: gcr.io/mlops-new-447207/ml-project:latest
        ports:
        - containerPort: 5000
---
apiVersion: v1
kind: Service
metadata:
  name: ml-app-service
spec:
  type: LoadBalancer
  selector:
    app: ml-app
  ports:
  - protocol: TCP
    port: 80
    targetPort: 5000

🔁 Step 7: Jenkinsfile for CI/CD
pipeline {
    agent any
    stages {
        stage('Build Docker Image') {
            steps {
                sh 'docker build -t gcr.io/mlops-new-447207/ml-project:latest .'
            }
        }
        stage('Push to GCR') {
            steps {
                sh 'gcloud auth configure-docker'
                sh 'docker push gcr.io/mlops-new-447207/ml-project:latest'
            }
        }
        stage('Deploy to GKE') {
            steps {
                sh 'kubectl apply -f deployment.yaml'
            }
        }
    }
}

🧠 ML Pipeline Flow

graph TD;
    A[Raw Anime Data] --> B[Data Ingestion]
    B --> C[Data Processing]
    C --> D[Model Training]
    D --> E[Prediction]
    E --> F[Web Output]

🖥️ Web Interface
Hosted on GKE using Flask
User inputs their ID ➝ returns anime recommendations

👨‍💻 Author
Jayesh Pramod Patil
📍 Pune, Maharashtra
📧 patiljayesh6908@gmail.com
📞 +91-9096380075
🌐 GitHub | LinkedIn

<p align="center"> <em>“Train like Goku. Deploy like Bulma.”</em><br> <img src="https://media.giphy.com/media/Nx0rz3jtxtEre/giphy.gif" width="200px"/> </p> ```



