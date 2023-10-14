# DBSherlock (Python)
This is a Python implementation of DBSherlock: A Performance Diagnostic Tool for Transactional Databases (SIGMOD 2016)

## Environment Setup
Start docker container using docker compose, and login to the container
```bash
docker compose up -d
```
Install python packages
```bash
pip install -r requirements.txt
```

## Prepare Dataset
You will need to download DBSherlock dataset and convert it to json format.
### Download DBSherlock Dataset
Download TPCC 16w dataset
```bash
wget -P data/original_dataset/ https://github.com/dongyoungy/dbsherlock-reproducibility/raw/master/datasets/dbsherlock_dataset_tpcc_16w.mat
```
Download TPCC 500w dataset
```bash
wget -P data/original_dataset/ https://github.com/dongyoungy/dbsherlock-reproducibility/raw/master/datasets/dbsherlock_dataset_tpcc_500w.mat
```
Download TPCE 3000 dataset
```bash
wget -P data/original_dataset/ https://github.com/dongyoungy/dbsherlock-reproducibility/raw/master/datasets/dbsherlock_dataset_tpce_3000.mat
```

### Data Convertion
Convert TPCC 16w dataset to json format
```bash
python scripts/data/convert_data.py --input data/original_dataset/dbsherlock_dataset_tpcc_16w.mat \
--out_dir data/converted_dataset \
--prefix tpcc_16w
```
Convert TPCC 500w dataset to json format
```bash
python scripts/data/convert_data.py --input data/original_dataset/dbsherlock_dataset_tpcc_500w.mat \
--out_dir data/converted_dataset \
--prefix tpcc_500w
```
Convert TPCE 3000 dataset to json format
```bash
python scripts/data/convert_data.py --input data/original_dataset/dbsherlock_dataset_tpce_3000.mat \
--out_dir data/converted_dataset \
--prefix tpce_3000
```
### How to load the dataset in Python
Please refer to [src/data/README.md](src/data/README.md)

## Run Experiments
### Experiment 1
```bash
python scripts/experiments/exp1.py
```