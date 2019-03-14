# CUDA

## 1. Install

```bash
sodo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update
sodo apt-get install nvidia-367 nvidia-settings
```

https://zhuanlan.zhihu.com/p/28954367

https://stackoverflow.com/questions/41409842/ubuntu-16-04-cuda-8-cuda-driver-version-is-insufficient-for-cuda-runtime-vers

[How to complete remove CUDA](https://askubuntu.com/questions/530043/removing-nvidia-cuda-toolkit-and-installing-new-one)

update kernel

```bash
uname -r
sudo apt-get purge linux-image-4.[].[]-[]-generic
sudo apt-get purge linux-headers-4.[].[]-[]-generic

sudo update-initramfs -u
```



```bash
sudo update-initramfs -u
lsmod | grep nouveau
```



```bash
sudo service lightdm stop
```





