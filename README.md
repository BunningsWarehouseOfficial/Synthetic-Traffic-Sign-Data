# Synthetic Dataset Generation for Traffic Sign Damage Detection

🚧 *Work in progress.*

## Abstract
**From Alexandros Stergiou's original [repository](https://github.com/alexandrosstergiou/Traffic-Sign-Recognition-basd-on-Synthesised-Training-Data) and [paper](https://www.mdpi.com/2504-2289/2/3/19):**
> A main concern of road sign recognition systems has been the use of a large dataset obtained through research and manually finding sufficient data in order to conduct the necessary experiments and optimally, create a model with high accuracy rates. The purpose of this dissertation is the introduction of a traffic sign recognition system, which is based on synthetic data generation methods (SGTD), that is not bound to a specific set of road signs as occurring in a country or region, but can generate training examples given a defined set of templates. This work can significantly decrease the time spent by researches or system developers in order to acquire data and allows the reallocation of the main project focus on the construction of the predictive model. Although, the training dataset is synthetically created, the variations in visual appearance between examples, such as differences in illumination and rotation, are reflected on each image to best represent road signs in real world settings and have been implemented using Computer Vision methods. The main classifier that the system is based on is a Convolutional Neural Network (CNN) that can achieve close to state-of-the art accuracy results when trained on actual data.

We have modified Stergiou's code to create synthetic data placed over a full background image to provide holistic data that can be used to train a detection model directly rather than training for simple classification. Furthermore, we have implemented various kinds of synthetic damage that are applied to the signs. This damage is quantified by using the percentage of the sign that has been changed or obscured. The purpose of this is to train a detection model to not only detect each class of sign, but to also detect the class and severity of damage that is present on the sign, if any.
 
 ![damaged_examples](https://github.com/BunningsWarehouseOfficial/Traffic-Sign-Damage-Detection-using-Synthesised-Training-Data/blob/main/Figures/Damaged_examples.png "Templates")
 
The above shows some close-up examples of damage applied to signs.

## Installation
Git is required to download and install the repo. If on Linux or Mac, open Terminal and follow these commands:
```sh
$ sudo apt-get update
$ sudo apt-get install git
$ git clone https://github.com/ai-research-students-at-curtin/Traffic-Sign-Damage-Detection-using-Synthesised-Training-Data.git
```

Using a Anaconda environment is recommended. Install Miniconda (terminal only) [here](https://docs.conda.io/en/latest/miniconda.html). Once installed, run the following command while in the repo directory:
```sh
$ conda env create -f environment.yaml
```

*Alternatively*, to install the required packages using pip, simply execute the following command (`pip3` may be exchanged with `pip`):
```sh
$ pip3 install -r requirements.txt
```

[comment]: <> (Note that the synthetic dataset SGTSD will need aprox. 10GB and the sample set used for training will be close to 1GB.)


## Usage
Navigate to the `signbreaker` directory. To run the generator, simply execute the following command:
```sh
$ python create_dataset.py
```
The `--output_dir` argument can be used to specify a custom directory for the generated dataset. A complete path is required.

The generator can be configured with by modifying the `config.yaml` file.

Please allow for sufficient storage space when running the dataset generator. With the default config values and the below example inputs, the size of the dataset will be on the order of 50-100 GB.

**Example traffic sign templates** for data generation can be downloaded [here](https://drive.google.com/file/d/1a65EPNK_3gNII8VXBKDZ0N6H_mL6F6P1/view?usp=sharing).

**Example backgrounds** for data generation can be downloaded [here](https://drive.google.com/file/d/1WCfWVruL0_WxnMaYJ-qzQD0cnFO478fh/view?usp=sharing).

## License
MIT

## Contributors
Kristian Rados *(Primary)* \
kristian.rados@student.curtin.edu.au

Seana Dale \
seana.dale@student.curtin.edu.au

Allen Antony \
allenantony2001@gmail.com

Jack Downes \
jack.downes@postgrad.curtin.edu.au
