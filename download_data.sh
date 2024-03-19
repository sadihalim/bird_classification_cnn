#!/bin/bash

# Configure Kaggle API
mkdir /root/.kaggle
echo "{\"username\":\"${KAGGLE_USERNAME}\",\"key\":\"${KAGGLE_KEY}\"}" > /root/.kaggle/kaggle.json
chmod 600 /root/.kaggle/kaggle.json

# Download the dataset
kaggle datasets download -d gpiosenka/100-bird-species -p /data

# Keep the container running so that the data can be copied out
tail -f /dev/null
