
# BEVFusion Framework Setup and Execution

This repository provides setup instructions and usage guidelines for running the BEVFusion framework for **3D Object Detection using LiDAR and Multi-view Camera Fusion in the Bird's Eye View** inside Docker. It is based on the [original BEVFusion repository by MIT HAN Lab](https://github.com/mit-han-lab/bevfusion).

---

## Docker container setup

Created and re-used a named Docker container (**bevfusion**) for convenient integration and consistent use inside Visual Studio Code.

---

## First-time user setup

### 1. Build the Docker image

Run this command only if the Dockerfile has been updated:

```bash
docker build -t bevfusion_original .
```

### 2. Create and run a named Docker container

```bash
docker run --gpus all -it \
  --name bevfusion-original \
  -v "/home/prabuddhi/bevfusion_final:/home/bevfusion" \
  -v "/media/prabuddhi/Crucial X91/bevfusion-main/data/nuscenes:/dataset" \
  --shm-size=16g \
  bevfusion_original /bin/bash
```

* Replace the path with the actual dataset directory path on the host machine.
* You can clone this repository from [https://github.com/Prabuddhi-05/bevfusion_final.git](https://github.com/Prabuddhi-05/bevfusion_final.git).
* Attach to this container via VS Code:
  * Ctrl + Shift + P → Attach to an existing container

### 3. Initial setup inside the container

Run the following commands:

```bash
cd /home
cd bevfusion
python setup.py develop # Install BEVFusion in development mode
mkdir -p data
ln -s /dataset ./data/nuscenes # Create a symbolic link to connect the dataset on the host machine to Docker
```

### 4. Fix NumPY related error

**NumPy attributeError**:

* Downgrade NumPy to resolve attribute errors:

```bash
conda install numpy=1.23.5 -y
```

### 5. Create swap memory (Prevents crashes) (Optional)

* Check memory usage (optional):

```bash
htop
free -h
```

* Create a 64 GB swap file:

```bash
sudo fallocate -l 64G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
sudo bash -c "echo '/swapfile none swap sw 0 0' >> /etc/fstab"
```

### 6. Data preprocessing (Optional)

* Edit the preprocessing script to skip unnecessary steps:
* Comment out create_groundtruth_database(...) in `/home/bevfusion/tools/create_data.py`

### 7. Run preprocessing:

```bash
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0
```

### 8. Download pre-trained weights

```bash
./tools/download_pretrained.sh
```

### 9. Run evaluation

```bash
torchpack dist-run -np 1 python tools/test.py \
  configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
  pretrained/bevfusion-det.pth --eval bbox
```

### 10. Exit the container (While progress is being saved)

```bash
exit
```

## Outputs

The model evaluates:

* **3D object detection** using fused **6-camera and LiDAR inputs**.
* Metrics include **NDS**, **mAP**, error metrics, and per-class results.

---

# **Additional setup for Training**

Run the following additional steps to fix common issues:


### **1. Install yapf**
```bash
pip install yapf==0.30.0
```


### **2. Set PyTorch distributed environment**
```bash
export MASTER_HOST=localhost
export MASTER_PORT=29500
```


### **3. Fix TensorBoard import issue**

Edit:
```bash
nano /opt/conda/envs/bevfusion/lib/python3.8/site-packages/torch/utils/tensorboard/__init__.py
```

Replace content with:
```python
import tensorboard
try:
    from setuptools._distutils.version import LooseVersion
except ImportError:
    from distutils.version import LooseVersion  # fallback for older setups

if not hasattr(tensorboard, '__version__') or LooseVersion(tensorboard.__version__) < LooseVersion('1.15'):
    raise ImportError('TensorBoard logging requires TensorBoard version 1.15 or above')

del LooseVersion
del tensorboard

from .writer import FileWriter, SummaryWriter  # noqa: F401
from tensorboard.summary.writer.record_writer import RecordWriter  # noqa: F401
```


### **4. Fix MMCV version**
```bash
pip uninstall mmcv-full -y
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
```


# **Run Training**

To start training with pre-trained checkpoints:
```bash
torchpack dist-run -np 1 python tools/train.py \
  configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
  --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
  --load_from pretrained/lidar-only-det.pth
```
## Subsequent runs (Reuse container)

* Restart and reuse your named container without data loss:

```bash
docker start -ai bevfusion-original
```

* You can directly re-run evaluations or training as required inside this container.
---

For detailed framework documentation, visit [BEVFusion GitHub](https://github.com/mit-han-lab/bevfusion).
