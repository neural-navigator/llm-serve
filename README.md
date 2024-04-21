# LLM Deployment Project

This project aims to deploy a Language Model (LLM) using Docker with NVIDIA GPU support. Below are the steps to set up and deploy the model.

## Steps:

1. **Grant Docker Access to NVIDIA GPU:**
   - Follow the provided link to install necessary components for Docker to utilize the system's NVIDIA GPU. This step is crucial for enabling GPU acceleration within the Docker environment.
   - Link: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
  
2. **Configure Model:**
   - Open `constants.py` and change the model name to specify the desired LLM to run. This allows flexibility in choosing different models for deployment.
  
3. **Set Hugging Face Access Token:**
   - Create a `.env` file and set your Hugging Face access token. This token is required for accessing models from the Hugging Face Model Hub.
  
4. **Build Docker Image:**
   - Run the command `docker build -t <image_name> .` to build the Docker image. This image will contain all necessary dependencies and configurations for deploying the LLM.
  
5. **Run Docker Container:**
   - Launch the Docker container with NVIDIA runtime support using the command:
     ```
     docker run -p 8000:8000 --runtime=nvidia --gpus all <image_name>
     ```
   - Replace `<image_name>` with the name of the Docker image built in the previous step.
   - This command exposes port 8000 to interact with the deployed LLM.

## Additional Notes:
- Ensure Docker is properly installed on your system before proceeding.
- Make sure NVIDIA GPU drivers are installed and up-to-date.
- Verify that the Hugging Face access token is valid and correctly set in the `.env` file.
- Run nvidia-smi to check whether GPU is being used or not
