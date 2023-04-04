# ICIP-MBG-PATREO
Entry to ICIP Grand Challenge 2023 - Automatic Detection of Mosquito Breeding Grounds

## Environment Setup
This setup adapts the set of recommendations from the original [yolo repository](https://github.com/WongKinYiu/yolov7)

```python
# create the docker container, you can change the share memory size if you have more.
docker run --gpus all --name yolov7_mbg -it -v dataset_path/:/dataset/ -v code_path/:/src --shm-size=2g nvcr.io/nvidia/pytorch:21.08-py3

# apt install required packages
apt-get update

# install required packages
pip install pandas lxml 

# go to code folder
cd /src
```

## How to run

### Formatting dataset
Run ```mbg2yolo.py``` referencing the path containing **only test samples**. The script will convert original xml annotations into YOLO formatting. Assuming the docker enviroment suggested above, the dataset should be located at ```/dataset```.   

For example:
```
python mbg2yolo.py --path /dataset/test/
````

> Note that we are assuming folder structure as released by the challenge (```flight-mbgXY/ann/videoXY.xml```, ```flight-mbgXY/avi/videoXY.avi```) 

### Download trained weights

Weights file should be store inside folder "weights" of this repository. Assuming the docker environment suggested above, the weights should be located at ```/src/weights``` as follows: 
```
wget https://www.dropbox.com/s/9anb8hfsbyt9j6e/yolov7_mbg_adam_960_alltrain.pt -P /src/weights/
```

### Run model

Important parameters: 
- --batch-size: size of each image batch (default: 32)
- --device: cuda device, i.e. 0 or 0,1,2,3 or cpu
- --verbose: report mAP by class
- --project: root folder to save results (default: runs/test)
- --name: folder created to store results (within --project)

