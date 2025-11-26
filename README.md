# B-Spine service

# Usage
## Installation
Please run the following scripts to create a envoriment and install dependencies. 
```
conda create -n bspine python=3.8
conda activate bspine
pip install torch==1.8.0+cpu torchvision==0.9.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt

# download mmcv full
wget https://download.openmmlab.com/mmcv/dist/1.3.5/torch1.8.0/cpu/mmcv_full-latest%2Btorch1.8.0%2Bcpu-cp38-cp38-manylinux1_x86_64.whl
pip install ./mmcv_full-latest%2Btorch1.8.0%2Bcpu-cp38-cp38-manylinux1_x86_64.whl
```

## Service
Register this service as a system service for convenient startup at boot.
```
sudo cp ./rc.local /etc/rc.local
sudo chmod +x /etc/rc.local
systemctl start rc-local.service
```