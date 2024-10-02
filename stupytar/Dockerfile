# Dockerfile
FROM jupyter/scipy-notebook:x86_64-ubuntu-22.04

# Install any additional packages you want
RUN pip install imblearn torch torchaudio torchvision

# You can also set environment variables here if needed
# ENV JUPYTER_TOKEN='your_password_here'

# Expose port 8888 for Jupyter Notebook
EXPOSE 8888
