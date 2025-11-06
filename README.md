### 🛠 Step-by-Step Deployment Guide for Financial-Transactions-Fraud-Detection-Project on WSL
👉 We can deploy our *Financial_Transactions_Fraud_Detection* project on a local WSL (Windows Subsystem for Linux) machine by cloning the GitHub repository, building the Docker image, and running the container within WSL. The key steps are: **install Docker in WSL, clone the repo, build the image, and run the container with mapped ports for API access.**

👉 Since our repository is already structured for Docker, the main effort is just **building and running the container inside local WSL**. 

---
#### 1. ✅ Prepare WSL Environment
- Make sure we have **WSL2** installed and a Linux distribution (Ubuntu recommended).  
- Update packages:
  ```bash
  sudo apt update && sudo apt upgrade -y
  ```

#### 2. 🐳 Install Docker in WSL
- Install Docker:
  ```bash
  sudo apt install docker.io -y
  ```
- Enable and start Docker service:
  ```bash
  sudo service docker start
  ```
- Add Our user to the Docker group (so you do not need `sudo` every time):
  ```bash
  sudo usermod -aG docker $USER
  newgrp docker
  ```

#### 3. 📂 Clone Your Repository
Inside WSL, run:
```bash
git clone https://github.com/Neang-Chhong-Makara/Financial_Transactions_Fraud_Detection.git
cd Financial_Transactions_Fraud_Detection
```

#### 4. 🏗 Build Docker Image
- If our repository has a `Dockerfile`, build the image:
  ```bash
  docker build -t financial_transactions_traud_detection_app .
  ```
- If we use **Docker Compose**, run:
  ```bash
  docker-compose up --build -d
  ```

#### 5. 🚀 Run the Container
- Run the container and expose ports (e.g., 5000 for API):
  ```bash
  docker run -d -p 5000:5000 financial_transactions_traud_detection_app
  ```
- This maps container port `5000` to our local machine’s port `5000`.

#### 6. 🌐 Access the Application
- Open our browser or use `curl`:
  ```bash
  curl http://localhost:5000
  ```
- If the project exposes an API endpoint (e.g., `/predict`), we can send JSON transaction data for fraud detection.

#### 7. 📊 Verify MLflow / Dashboard (Optional)
- If MLflow tracking is included, expose its port (default `5000` or `5001`):
  ```bash
  docker run -d -p 5001:5001 financial_transactions_traud_detection_app
  ```
- Then access MLflow UI at `http://localhost:5001`.

---

### 🔑 Key Notes
- **Consistency:** Docker ensures the same environment across dev and production.  
- **Scalability:** We can later deploy this container to Kubernetes or the cloud.  
- **Auditability:** MLflow and artifact logging inside Docker make fraud detection experiments reproducible.  

---
