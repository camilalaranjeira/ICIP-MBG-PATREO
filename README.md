# ICIP-MBG-PATREO
Entry to ICIP Grand Challenge 2023 - Automatic Detection of Mosquito Breeding Grounds

## Environment Setup
This setup adapts the set of recommendations from the original [yolo repository](https://github.com/WongKinYiu/yolov7)

```python
# create the docker container, you can change the share memory size if you have more.
nvidia-docker run --name yolov7_mbg -it -v dataset_path/:/dataset/ -v code_path/:/src --shm-size=2g nvcr.io/nvidia/pytorch:21.08-py3

# apt install required packages
apt update
apt install -y zip htop screen libgl1-mesa-glx

# pip install required packages
pip install seaborn thop

# go to code folder
cd /yolov7_mbg
```

## How to run

Important parameters: 
- --batch-size: size of each image batch (default: 32)
- --device: cuda device, i.e. 0 or 0,1,2,3 or cpu
- --verbose: report mAP by class
- --project: root folder to save results (default: runs/test)
- --name: folder created to store results (within --project)

