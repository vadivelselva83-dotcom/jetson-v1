Linux 32 GB RAM Env
# jetson-v1
Object detection of headlights using jetson
# JetPack already includes CUDA/cuDNN, etc.
# (A) Install OpenCV with GStreamer support (usually in JetPack images):
python3 -c "import cv2; print(cv2.getBuildInformation())" | grep -i gstreamer

# (B) Install TensorFlow in NVIDIA’s L4T TensorFlow container (recommended):
sudo docker run --runtime nvidia -it --rm \
  --network host \
  -v $PWD:/workspace \
  nvcr.io/nvidia/l4t-tensorflow:r35.4.1-tf2-py3

# Inside the container:
python3 -m pip install --upgrade pip
pip install tensorflow-addons matplotlib pycocotools opencv-python tf-models-official==2.15.0
