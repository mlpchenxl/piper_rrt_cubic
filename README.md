# PiperPlan

## Setup

### 1. Prerequest (Optional)


> :warning: **Note :** 如果有python 3.10 的版本，cuda 11.8 以及对应的torch版本可以跳过(We only test on python 3.10)


#### 创建虚拟环境

```bash
conda create -n piper_planning_env python=3.10.13
```

#### Install cuda 11.8 (Follow these steps) :

``` bash
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run

sudo sh cuda_11.8.0_520.61.05_linux.run

vim ~/.bashrc

添加
export PATH=$PATH:/usr/local/cuda-11.8/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.8/lib64
```


#### 激活虚拟环境
 
```bash
conda activate piper_planning_env
```

#### 安装torch

```bash
pip install torch==2.2.1+cu118 torchvision==0.17.1+cu118 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118
```

### 2. Install (Must to do)
- Intsall pyroboplan
```bash
cd pyroboplan
pip install -e .
pip install numpy==1.24.0 (忽略cmeel-boost版本报错)
```

### 3. Example

```bash
cd ..
python python piper_example.py
```

