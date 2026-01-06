# CoRe-VL: From Concepts to Regions: Structured Knowledge Injection for Radiology Report Generation with Large Vision-Language Models

## 1. Prerequisites

Make sure your local environment has the following installed:

* `pytorch>=1.12.1 & <=1.9`
* `numpy == 1.15.1`
* `python >= 3.10`
* `peft`
* `transformers==4.30.2`
* `lightning==2.0.5`

**2. Prepare the training dataset**

IU-xray: download the dataset from official website.

Mimic-cxr: you can download  the dataset from [official website](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)

After downloading the data, place it in the ./data folder.

**3. How to MambaGen

### 1 Install IDE 

Our project is built on PyCharm Community Edition ([click here to get](https://www.jetbrains.com/products/compare/?product=pycharm-ce&product=pycharm)).

### 2 Environment setting
#### 2.1 Inpterpreter 
We recommend using `Python 3.10` or higher as the script interpreter. [Click here to get](https://www.python.org/downloads/release/python-3110/) `Python 3.10`. 
#### 2.2 Packages
Please follow the Prerequisites, utilize `pip install <package_name>` to construct the environment.
### 3 Train
Run `bash scripts/1-1.shallow_run_iuxray.sh` to train a model on the IU X-Ray dataset.

Run `bash bash scripts/4-1.shallow_run.sh` to train a model on the MIMIC-CXR dataset.


