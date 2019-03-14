# Conda

## 1. Install

## 2. Create New Repo

```bash
conda create -n sklearn python=3.5

conda-env list
```



## 3. Start Repo and Configuration

```bash
source activate sklearn
sudo apt-get install python3-pip
pip3 install numpy
pip3 install scipy
pip3 install opencv-python
pip3 install -U scikit-learn # or conda: conda install scikit-learn

conda install -c anaconda tensorflow=gpu

conda install ipython
conda install jupyter
jupyter notebook
```

## 4. Others

**Rename a environment**:

```bash
conda create --name [new_name] --clone [old_name]
conda remove --name [old_name] --all
```

