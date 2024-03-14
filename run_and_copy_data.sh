docker build -t downloader -f dataset_download.dockerfile .
mkdir data
docker run -v $(pwd)/data:/app/data downloader


