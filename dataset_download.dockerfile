# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Preparing the environement
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Make directory for Kaggle API key
RUN mkdir /root/.kaggle/

# Copy the Kaggle API key into the container
COPY kaggle.json /root/.kaggle/kaggle.json
RUN chmod 600 /root/.kaggle/kaggle.json
# RUN kaggle datasets list

# # Copy the rest of your application's code
COPY . .

# # Command to download dataset using Kaggle API
RUN kaggle datasets download -d gpiosenka/100-bird-species -p /usr/src/app/data --unzip

# Command to dowload it using python
# CMD ["python", "src/download_data.py", "gpiosenka/100-bird-species"]
