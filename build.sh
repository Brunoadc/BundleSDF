ROOT=$(pwd)
cd /kaolin && pip install -e .
cd ${ROOT}/mycuda && rm -rf build *egg* && pip install -e .
cd ${ROOT}/BundleTrack && rm -rf build && mkdir build && cd build && cmake .. && make -j11
pip install git+https://github.com/facebookresearch/segment-anything.git

pip uninstall --yes opencv-python
pip install opencv-python-headless pycocotools matplotlib onnxruntime onnx

pip install progressbar2

pip uninstall --yes scipy
pip install scipy==1.9