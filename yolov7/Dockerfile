FROM python:3.9-slim

WORKDIR /yolov7

# cài thư viện hệ thống cần thiết cho OpenCV
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 git && rm -rf /var/lib/apt/lists/*

# copy mã nguồn và weights
COPY . /yolov7

# cài thư viện Python
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir opencv-python numpy pandas pyyaml

# lệnh mặc định
CMD ["python", "detect.py", "--weights", "weights/yolov7.pt", "--source", "images", "--img-size", "640", "--conf", "0.25", "--save-txt", "--save-conf"]