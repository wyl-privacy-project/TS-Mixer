# TS-Mixer: An all-MLP Model via Token Shift
## TS-Mixer
![Figure 1. The overall architecture of the proposed TS-Mixer](https://github.com/wyl-privacy-project/TS-Mixer/blob/main/Figure/TS-Mixer.jpg)
## TS-MLP Block
![Figure 2. TS-MLP block](https://github.com/wyl-privacy-project/TS-Mixer/blob/main/Figure/TS_MLP%20BLock.jpg)
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
## Caching Vocab Hashes

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
The checkpoints used for evaluation are available [here]().
### Topic Classification 
|Model|AG News(%)|DBpedia(%)|Params(M)|
|:--:|:--:|:--:|:--:|
| XLNet | 95.55 | 99.40 | 240 |
| Bert Large | / | 99.36 | 340 |
| pNLP-Mixer-XS | 88.89 | 98.03 | 0.272 |
| pNLP-Mixer-Base | 90.00 | 98.33 | 1.3 |
| pNLP-Mixer-XL | 91.03 | 98.40 | 5.0 |
| TS-Mixer-S | 91.43 | 98.57 | 0.174 |
| TS-Mixer-S | 91.99 | 98.69 | 0.429 |
| TS-Mixer-S | 92.10 | 98.75 | 1.2 |

### Sentiment Analysis

| Model | IMDB(%) | Yelp-2(%) | Amazon-2(%) | Params(M) |
|:--:|:--:|:--:|:--:|:--:|
| RoBERTa | 95.3 | / | / | 125|
| XLNet | 96.8 | 98.63 | 97.89 | 240 |
| Bert Large | 95.49 |	98.11 |	97.37 | 340 | 
| BERT-ITPT-FiT | 95.63	| 98.08	| / | / |
| Longformer | 95.7	| /	| / | 149 |
| pNLP-Mixer-XS | 81.90	| 93.93	| 92.90 | 1.2/1.2/0.403 |
| pNLP-Mixer-Base | 78.60 |	94.41 |	93.83 |	2.1/2.1/1.4 |
| pNLP-Mixer-XL | 82.90 |	94.62 |	93.50	| 6.3/6.3/5.3 |
| TS-Mixer-S | 88.78	| 96.16 |	93.81	| 0.174 |
| TS-Mixer-B | 89.75	| 96.21	| 94.38 |	0.429 |
| TS-Mixer-L | 89.15 |	96.34	| 94.55 |	1.2 |

###  Natural Language Inference

| Model | SST-2(%) |	CoLA(%) |	QQP(%) | Params(M) |
|:--:|:--:|:--:|:--:|:--:|
| RoBERTa | 96.70	| 67.80	| 90.20| 125|
| XLNet | 97.00 |	70.20	| 90.40 | 240 |
| Bert Large | 94.90 |	60.50 |	72.10 | 340 | 
| TinyBert | 92.60	| 43.30 |	71.30 | 14.5 |
| MobileBert_Tiny | 91.70	| 46.70	| 68.90	| 15.1 |
| MobileBert | 92.80 |	50.50	| 70.20	| 25.3 |
| gMLP | 94.8	| / |	/ |	365 |
| pNLP-Mixer-XS | 79.74	| 70.04	| 83.70	| 0.206/0.174/0.272|
| pNLP-Mixer-Base | 78.89	| 69.69	| 84.45	| 1.3/1.2/1.3 |
| pNLP-Mixer-XL | 80.94	| 69.33	| 84.90	| 4.9/4.8/5.0 |
| HyperMixer | 80.90 |	/ |	83.70 |	11 |
| TS-Mixer-S | 83.74 |	70.09	| 82.53	| 0.174 |
| TS-Mixer-B | 84.76	| 70.19	| 83.76 |	0.429 |
| TS-Mixer-L | 84.92	| 69.93	| 84.58 |	1.2 |
