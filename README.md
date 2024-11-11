# MLOPS Project 2: Containerization

### Local Setup
##### 1. Clone the Repo
```bash
git clone https://github.com/yourusername/your-repo.git
cd your-repo
```
##### 2. Build the Docker image
```bash
docker build -t mlops_proj2 .
```

##### 3. Get Wandb API-Key

 - Create a account and generate key with: https://wandb.ai/settings#api

##### 4. Run Docker container

- Run with default parameters:
```bash
docker run -e WANDB_API_KEY="YOUR KEY" mlops_proj2
```

- Run with custom parameters:
```bash
docker run -e WANDB_API_KEY="YOUR KEY" -e LR=0.00002 -e WARMUP_STEPS=50 -e DECAY=0.00002 -e SAVE_PATH="model.pth" mlops_proj2
```
