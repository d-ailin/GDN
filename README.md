# GDN

Code implementation for : [Graph Neural Network-Based Anomaly Detection in Multivariate Time Series(AAAI'21)](https://arxiv.org/pdf/2106.06947.pdf)


# Installation
### Requirements
* Python >= 3.6
* cuda == 10.2
* [Pytorch==1.5.1](https://pytorch.org/)
* [PyG: torch-geometric==1.5.0](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

### Install packages
```
    # run after installing correct Pytorch package
    bash install.sh
```

### Quick Start
Run to check if the environment is ready
```
    bash run.sh cpu msl
    # or with gpu
    bash run.sh <gpu_id> msl    # e.g. bash run.sh 1 msl
```


# Usage
We use part of msl dataset(refer to [telemanom](https://github.com/khundman/telemanom)) as demo example. 

## Data Preparation
```
# put your dataset under data/ directory with the same structure shown in the data/msl/

data
 |-msl
 | |-list.txt    # the feature names, one feature per line
 | |-train.csv   # training data
 | |-test.csv    # test data
 |-your_dataset
 | |-list.txt
 | |-train.csv
 | |-test.csv
 | ...

```

### Notices:
* The first column in .csv will be regarded as index column. 
* The column sequence in .csv don't need to match the sequence in list.txt, we will rearrange the data columns according to the sequence in list.txt.
* test.csv should have a column named "attack" which contains ground truth label(0/1) of being attacked or not(0: normal, 1: attacked)

## Run
```
    # using gpu
    bash run.sh <gpu_id> <dataset>

    # or using cpu
    bash run.sh cpu <dataset>
```
You can change running parameters in the run.sh.

# Others
SWaT and WADI datasets can be requested from [iTrust](https://itrust.sutd.edu.sg/)


# Citation
If you find this repo or our work useful for your research, please consider citing the paper
```
@inproceedings{deng2021graph,
  title={Graph neural network-based anomaly detection in multivariate time series},
  author={Deng, Ailin and Hooi, Bryan},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={5},
  pages={4027--4035},
  year={2021}
}
```
