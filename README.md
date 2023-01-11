# TS-Mixer: An all-MLP Model via Token Shift
## TS-Mixer
![Figure 1. The overall architecture of the proposed TS-Mixer](https://github.com/wyl-privacy-project/TS-Mixer/tree/main/Figure/TS-Mixer.png)
## TS-MLP Block
![Figure 2. TS-MLP block](https://github.com/wyl-privacy-project/TS-Mixer/blob/main/Figure/TS_MLP%20BLock.png)
## Usage
### Install
- Clone this repo:
```bash
git clone https://github.com/wyl-privacy-project/TS-Mixer
cd TS-Mixer
```
- Create a conda virtual environment and activate it:
```bash
conda create -n TSMixer python=3.8 -y
conda activate TSMixer
```
## Caching Vocab Hashes(like pNLP-Mixer)

```bash
python projection.py -v=wordpiece/vocab.txt -c=cfg/Config_Path -o=OutPut_File
```
- Config_Path: path to the configurations file
- OutPut_File: path where the resulting file will be saved,default='/vocab.npy')
## Train/Test

```bash
python run.py -c=Config_Path -n=MODEL_NAME -m=MODE -p=CKPT_PATH
```
- Config_Path: path to the configurations file
- MODEL_NAME: model name to be used for pytorch lightning logging
- MODE: train or test
- CKPT_PATH: checkpoint path to resume training from or to use for testing

## Experimental Results
The checkpoints used for evaluation are available [here](https://drive.google.com/drive/folders/1wtnWHfNjO9p0sR95M8W4avhqFUnZHooS?usp=sharing).
### Topic Classification 
|Model|AG News(%)|DBpedia(%)|Params(M)|
|:--:|:--:|:--:|:--:|
| TS-Mixer-S | 91.43 | 98.57 | 0.174 |
| TS-Mixer-S | 91.99 | 98.69 | 0.429 |
| TS-Mixer-S | 92.10 | 98.75 | 1.2 |

### Sentiment Analysis

| Model | IMDB(%) | Yelp-2(%) | Amazon-2(%) | Params(M) |
|:--:|:--:|:--:|:--:|:--:|
| TS-Mixer-S | 88.78	| 96.16 |	93.81	| 0.174 |
| TS-Mixer-B | 89.75	| 96.21	| 94.38 |	0.429 |
| TS-Mixer-L | 89.15 |	96.34	| 94.55 |	1.2 |

###  Natural Language Inference

| Model | SST-2(%) |	CoLA(%) |	QQP(%) | Params(M) |
|:--:|:--:|:--:|:--:|:--:|
| TS-Mixer-S | 83.74 |	70.09	| 82.53	| 0.174 |
| TS-Mixer-B | 84.76	| 70.19	| 83.76 |	0.429 |
| TS-Mixer-L | 84.92	| 69.93	| 84.58 |	1.2 |
