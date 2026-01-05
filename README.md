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

IU-xray: download the dataset from [here](https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg/view)

Mimic-cxr: you can download our preprocess annotation file from [here](https://drive.google.com/file/d/14689ztodTtrQJYs--ihB_hgsPMMNHX-H/view?usp=sharing) and download the images from [official website](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)

After downloading the data, place it in the ./data folder.
