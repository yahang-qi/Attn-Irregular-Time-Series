# Temporal Graphical Modelling

## Description

42 variables -- 6 of them are general descriptors, and the remainder are time series

Records from 12,000 ICU stays. Records are split into 3 subsets of the same size



**General Descriptors**

- *RecordID* (a unique integer for each ICU stay)
- *Age* (years)
- *Gender* (0: female, or 1: male)
- *Height* (cm)
- *ICUType* (1: Coronary Care Unit, 2: Cardiac Surgery Recovery Unit, 3: Medical ICU, or 4: Surgical ICU)
- *Weight* (kg)

**Time Series**

These 37 variables may be observed once, more than once, or not at all in some cases:



## Setup Environment
```bash
conda create -n TGM python=3.8.5
conda activate TGM
conda install pytorch==1.11.0 cudatoolkit=11.3 -c pytorch
pip install numpy (1.24.4)
pip install pandas
```





## Preprocess data

```bash
python ./tools/data_preprocessing.py --path <path-to-dataset> --set <subset-to-process> --output <directory-to-save-results>
```
Example:
```bash
python ./tools/data_preprocessing.py --path /home/qqqq/datasets/PhysioNet2012/ --set set-b --output /home/qqqq/datasets/PhysioNet2012_Preprocessed/
```
