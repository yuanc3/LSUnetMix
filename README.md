# LSUnetMix
We proposed a LSUnetMix modle which contains 3 modules: CIT (Channel Information Transmission), PE (Prospect Enhancement), and MDC (Multiscale Dilated Convolution).

## Requirements

Install from the ```requirement.txt``` using:
```angular2html
pip install -r requirements.txt
```

## Usage


### 1. Data Preparation
#### 1.1. GlaS and MoNuSeg Datasets
The original data can be downloaded in following links:
* MoNuSeG Dataset - [Link (Original)](https://monuseg.grand-challenge.org/Data/)
* GLAS Dataset - [Link (Original)](https://warwick.ac.uk/fac/cross_fac/tia/data/glascontest)
* DRIVE Dataset - [Link (Original)](https://drive.grand-challenge.org/)

Then prepare the datasets in the following format for easy use of the code:
```angular2html
├── datasets
    ├── DRIVE
    │   ├── Val_Folder
    │   │   ├── img
    │   │   └── labelcol
    │   └── Train_Folder
    │       ├── img
    │       └── labelcol
    ├── GlaS
    │   ├── Val_Folder
    │   │   ├── img
    │   │   └── labelcol
    │   └── Train_Folder
    │       ├── img
    │       └── labelcol
    └── MoNuSeg
        ├── Val_Folder
        │   ├── img
        │   └── labelcol
        └── Train_Folder
            ├── img
            └── labelcol
```
### 2. Training

First, change the settings in ```Config.py```, all the configurations including learning rate, batch size and etc. are in it.

Run:
```angular2html
python train_model.py
```

### 3. Testing
#### 3.1. Test the Model and Visualize the Segmentation Results
First, change the model_path and set the relative dataset in ```Config.py```.
Then run:
```angular2html
python test_model.py
```
You can get the Dice and IoU scores and the visualization results. 
#### 3.2. The pre-trained models of us.
We provide the best models in in our paper, and here is the links.

* MoNuSeG model - [Link (Original)](https://drive.google.com/file/d/1AQTqizlzSY0ljFr2oYBUMdaTk6e7lj3N/view?usp=sharing)
* GLAS model - [Link (Original)](https://drive.google.com/file/d/1YemmVw44lCDNBTYAhOmDx3LimxWIsT6m/view?usp=sharing)
* GDRIVE model - [Link (Original)](https://drive.google.com/file/d/1rWJp-Y2IRQ6wPTCRVZhCWcmU8U6YZI6V/view?usp=sharing)