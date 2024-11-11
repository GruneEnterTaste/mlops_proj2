# Use Python 3.10.12 as the base image
FROM python:3.10.12-slim

# Set the working directory inside the container
WORKDIR /app

# Upgrade pip and install dependencies
RUN python -m pip install --upgrade pip
RUN python -m pip install torch transformers lightning datasets wandb evaluate ipywidgets scikit-learn

# Copy local files to the container
COPY . .

# Run the main setup.py script
CMD ["python", "setup.py"]
