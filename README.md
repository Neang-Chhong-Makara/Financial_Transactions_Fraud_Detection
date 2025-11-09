### 🛠 Step-by-Step Deployment Guide for Financial-Transactions-Fraud-Detection-Project on local WSL
👉 To deploy the *Financial_Transactions_Fraud_Detection* project on a local WSL (Windows Subsystem for Linux) environment, we will clone the GitHub repository and use **Docker Compose** to build the image and run the container directly within WSL. 

👉 The key steps are: **install Docker and Docker Compose in WSL, clone the repository, and run the container with the appropriate port mappings for API and UI access.**

👉 Since the repository is already Docker-ready, the main task is simply **building and running the container inside the local WSL environment using Docker Compose.**

#### 1. ✅ Prepare WSL Environment
- Ensure **WSL2** is installed with a Linux distribution (Ubuntu recommended).  
- Update packages:
  ```bash
  sudo apt update && sudo apt upgrade -y
  ```

#### 2. 🐳 Install Docker and Docker Compose in WSL
- Install Docker and Docker Compose:
  ```bash
  sudo apt install docker.io docker-compose -y
  ```
- Enable and start Docker service:
  ```bash
  sudo service docker start
  ```
- Add the user to the Docker group (so `sudo` is not required every time):
  ```bash
  sudo usermod -aG docker $USER
  newgrp docker
  ```

#### 3. 📂 Clone Repository & Pull LFS Files Inside Local WSL, run:
- Ensure Git and Git LFS are installed:
```bash
sudo apt update
sudo apt install git-lfs -y
git lfs install
```
- Clone the repository and pull large files:
```bash
git clone https://github.com/Neang-Chhong-Makara/Financial_Transactions_Fraud_Detection.git
cd Financial_Transactions_Fraud_Detection
git lfs pull
```

#### 4. 🏗 Run Docker Compose
- Build and start containers in detached mode:
  ```bash
  docker-compose up --build -d
  ```

#### 5. 🚀 Verify Containers
- Check running containers:
  ```bash
  docker ps
  ```
- View logs if needed:
  ```bash
  docker-compose logs -f
  ```
- This confirms that the Streamlit and MLflow services are running correctly.

#### 6. 🌐 Access the Application
- Open a browser and access:
  - Streamlit UI → `http://localhost:8501`  
  - MLflow UI → `http://localhost:5000`  

### 🔑 Key Notes
- **Consistency:** Docker Compose ensures the same environment across development and production.  
- **Scalability:** Containers can later be deployed to Kubernetes or cloud platforms.  
- **Auditability:** MLflow and artifact logging inside Docker make fraud detection experiments reproducible. 
