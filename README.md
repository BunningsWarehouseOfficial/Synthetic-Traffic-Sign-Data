# Synthetic Dataset Generation for Traffic Sign Damage Detection

🚧 *Work in progress.*

## Abstract
- [ ] TODO: Add links to and make reference to damage assessment paper and also mention upcoming paper in the works

We have modified [code](https://github.com/alexandrosstergiou/Traffic-Sign-Recognition-basd-on-Synthesised-Training-Data) by Alexandros Stergiou ([2018 paper](https://www.mdpi.com/2504-2289/2/3/19)) to create synthetic traffic signs which are placed over a background image to create a holistic data that can be used to train a detection model directly rather than training for simple classification. Furthermore, we have implemented various kinds of synthetic damage that are applied to the signs. This damage is quantified by using the percentage of the sign that has been changed or obscured. The purpose of this is to train a detection model to not only detect each class of sign, but to also detect the class and assess the severity of damage that is present on the sign, if any.
 
 ![damaged_examples](https://github.com/BunningsWarehouseOfficial/Traffic-Sign-Damage-Detection-using-Synthesised-Training-Data/blob/main/Figures/Damaged_examples.png "Templates")
 
The above shows some close-up examples of damage applied to signs.

## Installation
Git is required to download and install the repo. If on Linux or Mac, open Terminal and follow these commands:
```sh
$ sudo apt-get update
$ sudo apt-get install git
$ git clone https://github.com/ai-research-students-at-curtin/Traffic-Sign-Damage-Detection-using-Synthesised-Training-Data.git
```

- [ ] TODO: Fix `environment.yaml`

Using an Anaconda environment is recommended. Install Miniconda (terminal only) [here](https://docs.conda.io/en/latest/miniconda.html). Once installed, run the following command while in the repo directory:
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

The generator can be configured by modifying the `config.yaml` file.

Please allow for sufficient storage space when running the dataset generator. With the default config values and the below example inputs, the size of the dataset will be on the order of 1-100 GB depending on the inputs and config.

**Example traffic sign templates** for data generation can be downloaded below. Place these templates into the `signbreaker/Sign_Templates/1_Input` directory. \
[[download]](https://drive.google.com/file/d/1kMAPRSOs9RqAtQu6-fUEn1fqkazIC3Kt/view?usp=sharing) | GTSDB | classes matched using [German](https://en.wikipedia.org/wiki/Road_signs_in_Germany) Wikipedia templates sourced using `wikipedia_sign_templates.py`, with extra speed limit templates created manually using SVG editing software. Covers 43/43 classes. \
[[download]](https://drive.google.com/file/d/1dALYTwtGMGrEXROh8KWBdLzH2_1Jxzmu/view?usp=sharing) | GTSDB | [UK](https://www.gov.uk/guidance/traffic-sign-images) templates used by Stergiou et al. in their paper. \
[[download]](https://drive.google.com/file/d/19_muDfADDh83zwIndZE3bsfbFh9KrGKD/view?usp=sharing) | GTSDB | classes matched as closely as possible using [UK](https://www.gov.uk/guidance/traffic-sign-images) templates. Covers ~31/43 classes.

**Example backgrounds** for data generation can be downloaded below. \
[[download]](https://drive.google.com/file/d/1WCfWVruL0_WxnMaYJ-qzQD0cnFO478fh/view?usp=sharing) | GTSDB | 4 GTSDB background images for quick testing. \
[[download]](https://drive.google.com/file/d/1dWkyX9-lGEE59odbthu3zFdZT9ksQ2nS/view?usp=sharing) | GTSDB | all 600 images from the GTSDB training set. \
[[download]](https://drive.google.com/file/d/1LvKXLakMttnXL7w4R3dl-dgmkv59cpQK/view?usp=sharing) | GTSDB | UK Google Images backgrounds used by Stergiou et al. in their paper, suitable only for creating image patch datasets for recognition tasks.

## License
- [ ] TODO: *Properly* update LICENSE file (sublicense?), then delete this heading

MIT

## Contributors
Kristian Radoš *(Primary)* \
kristianrados40@gmail.com
kristian.rados@student.curtin.edu.au

Seana Dale \
seana.dale@student.curtin.edu.au

Allen Antony \
allenantony2001@gmail.com

Prasanna Asokan \
prasanna.asokan@student.curtin.edu.au

Jack Downes \
jack.downes@postgrad.curtin.edu.au

## Citation
- [ ] TODO: Add damage assessment paper bibtex citation once available
```
```
- [ ] TODO: Add synthetic dataset paper bibtex citation once available
```
```
