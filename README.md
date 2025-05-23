# PiperPlan

## Setup

### 1. Prerequest (Optional)


> :warning: **Note :** 如果有python 3.10 的版本，cuda 11.8 以及对应的torch版本可以跳过(We only test on python 3.10)


#### 创建虚拟环境

```bash
conda create -n piper_planning_env python=3.10.13
```

#### 激活虚拟环境
 
```bash
conda activate piper_planning_env
```

#### 安装torch

```bash
pip install torch==2.2.1+cu118 torchvision==0.17.1+cu118 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118
```

### 2. Clone repo

```
git clone git@github.com:mlpchenxl/piper_rrt_cubic.git
cd piper_rrt_cubic
```

### 3. Install (Must to do)
- Intsall pyroboplan
```bash
cd pyroboplan
pip install -e .
pip install numpy==1.24.0 (忽略cmeel-boost版本报错)
```

### 4. Example

```bash
cd ..
python piper_example.py
```

## :heavy_exclamation_mark: Trouble Shotting
1. If you encounter the problem of: ```ImportError: /lib/x86_64-linux-gnu/libboost_python38.so.1.71.0: undefined symbol: _Py_fopen```

    To solve it, do:
    ```
    conda install pinocchio -c conda-forge # IF you could, do not using tsinghua source
    ```
2. TODO