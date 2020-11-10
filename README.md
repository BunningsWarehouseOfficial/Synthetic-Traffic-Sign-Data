# Utilising Synthetic Data Generation for the Detection of Damage in Traffic Signs

🚧 *Work in progress.*

## Abstract
**From Stergiou's original repository and paper:**
> A main concern of road sign recognition systems has been the use of a large dataset obtained through research and manually finding sufficient data in order to conduct the necessary experiments and optimally, create a model with high accuracy rates. The purpose of this dissertation is the introduction of a traffic sign recognition system, which is based on synthetic data generation methods (SGTD), that is not bound to a specific set of road signs as occurring in a country or region, but can generate training examples given a defined set of templates. This work can significantly decrease the time spent by researches or system developers in order to acquire data and allows the reallocation of the main project focus on the construction of the predictive model. Although, the training dataset is synthetically created, the variations in visual appearance between examples, such as differences in illumination and rotation, are reflected on each image to best represent road signs in real world settings and have been implemented using Computer Vision methods. The main classifier that the system is based on is a Convolutional Neural Network (CNN) that can achieve close to state-of-the art accuracy results when trained on actual data.

We have modified Stergiou's code to create synthetic data placed over a full background image to provide holistic data that can be used to train a detection model directly rather than training for simple classification. Furthermore, we have implemented various kinds of synthetic damage that are applied to the signs. This damage is quantified by using the percentage of the sign that has been changed or obscured. The purpose of this is to train a detection model to not only detect each class of sign, but to also detect the class and severity of damage that is present on the sign, if any.
 
 ![damaged_examples](https://github.com/BunningsWarehouseOfficial/Traffic-Sign-Damage-Detection-using-Synthesised-Training-Data/blob/main/Figures/Damaged_examples.png "Templates")
 
The above shows some close-up examples of damage applied to signs.

## Installation
Git is required to download and install the repo. You can open Terminal (for Linux and Mac) or cmd (for Windows) and follow these commands:
```sh
$ sudo apt-get update
$ sudo apt-get install git
$ git clone https://github.com/BunningsWarehouseOfficial/Traffic-Sign-Damage-Detection-using-Synthesised-Training-Data.git
```

[comment]: <> (Note that the synthetic dataset SGTSD will need aprox. 10GB and the sample set used for training will be close to 1GB.)
Please allow sufficient storage space when running the dataset generator.

## License
MIT

## Contact
Kristian Rados
(kristian.rados@student.curtin.edu.au)

Seana Dale
(seana.dale@student.curtin.edu.au)