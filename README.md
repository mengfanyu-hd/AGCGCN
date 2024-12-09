# Attention-Guided Clustered Graph Convolutional Network for Spatio-Temporal Traffic Forecasting

# Note

This repo is for the code implementation of our submitted paper **Attention-Guided Clustered Graph Convolutional Network for Spatio-Temporal Traffic Forecasting
**.

The readme file is updated as the libcity library is integrated now.

For the codes and details corresponding to our core contribution, please refer to the section [Anonymous Github version](https://github.com/mengfanyu-hd/AGCGCN)

If you find this repo useful, please cite it as follows,

# Quick Start

## Acknowledgements

We refer to the code implementation of [lib-city](https://bigscity-libcity-docs.readthedocs.io/en/latest/get_started/quick_start.html)
Please also cite the following papers if you find the code useful.

```latex
@inproceedings{libcity,
  author = {Wang, Jingyuan and Jiang, Jiawei and Jiang, Wenjun and Li, Chao and Zhao, Wayne Xin},
  title = {LibCity: An Open Library for Traffic Prediction},
  year = {2021},
  isbn = {9781450386647},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3474717.3483923},
  doi = {10.1145/3474717.3483923},
  booktitle = {Proceedings of the 29th International Conference on Advances in Geographic Information Systems},
  pages = {145–148},
  numpages = {4},
  keywords = {Spatial-temporal System, Reproducibility, Traffic Prediction},
  location = {Beijing, China},
  series = {SIGSPATIAL '21}
}
```


## 1. Prepare your dataset

You can create a new folder "raw_data" under the root path and download a dataset from the collection [libcity](https://bigscity-libcity-docs.readthedocs.io/en/latest/tutorial/install_quick_start.html#download-one-dataset) under the new path.

Then simply add the mapping matrix "XXX.mor.py" into the folder of a dataset e.g. ,  $ROOT_PATH/raw_data/METR_LA/METR_LA.mor.py. 

You can utilize our proposed mapping matrix or generate one by the provided utils.

## 2. Create a config file

A simple configs.json file to indicate some hyperparameters, e.g., the number of global nodes, hidden_size,$\eta_{1,2,3,4}$

```json
{
	"n1": 1,
	"n2": 1,
	"n3": 1,
	"n4": 1,
	"global_nodes":15,
	"nhid":32
}
```



## 3. Execution

Under the root path with the run_model.py, the program should be executed properly.

```bash
python3 ./run_model.py --task traffic_state_pred --model HIEST --config configs --dataset METR_LA
```


# Anonymous Github version

The following parts are organized as follows,

1. The model files
2. The processed adjacency matrices and mapping matrices for datasets.
3. The utils for solving BCC
The environment image preparation.



## 1. Model

Our model is under the path of ./code/HIEST.py.
We also provide an implementation of a Traffic-Transformer under the [guide of the lib-city](https://bigscity-libcity-docs.readthedocs.io/en/latest/developer_guide/implemented_models.html) 

These two models are for 'traffic-state-prediction', you can add them into the pipeline under the [instructions]((https://bigscity-libcity-docs.readthedocs.io/en/latest/developer_guide/implemented_models.html) ) provided by lib-city.

## 2. Processed Data

For the attributes self.adj_mx and self.Mor, they will be initialized with the processed adjacency matrix and mapping matrix. Please check the path settings to make it correspond with the dataset.

For the training datasets, you can refer to the [datasets collection of lib-city](https://bigscity-libcity-docs.readthedocs.io/en/latest/get_started/quick_start.html)

## 3. The utils for solving BCC

The utils for solving BCC are under the path of ./utils .

For the usage, you can refer to the visualization code under the path of ./code/visualization.py

## 4. Running environment

The running environment aligns with the [requirements of lib-city](https://github.com/LibCity/Bigscity-LibCity/blob/master/requirements.txt)

We are glad to share the following guide for build environment to ease reproducibility.

We implement the customized environment with [singularity](https://docs.sylabs.io/guides/3.7/user-guide/index.html) image for better execution.

If you are using Docker, the key idea should be similar with our implementation.

The singularity official documentation will provide the quick start-up with installation steps.

*All of the following scripts are executed on the **root path** of lib-city!*

