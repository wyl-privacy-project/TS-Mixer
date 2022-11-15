# TS-Mixer: An all-MLP Model via Token Shift
## TS-Mixer
![Figure 1. The overall architecture of the proposed TS-Mixer](https://github.com/wyl-privacy-project/TS-Mixer/blob/main/Figure/TS-Mixer.jpg)
## TS-MLP Block
![Figure 2. TS-MLP block](https://github.com/wyl-privacy-project/TS-Mixer/blob/main/Figure/TS_MLP%20BLock.jpg)
## Experimental Results
### Topic Classification 
<table align=center>
  <tr><th align=center>Model</th><th align=center>AG News(%)</th><th align=center>DBpedia(%)</th><th align=center>Params(M)</th></tr>
  <tr><td>XLNet</td><td>95.55</td><td>99.4</td><td>240</td></tr>
  <tr><td>Bert Large</td><td>/</td><td>99.36</td><td>340</td></tr>
  <tr><td>pNLP-Mixer-XS</td><td>88.89</td><td>98.03</td><td>0.272</td></tr>
  <tr><td>pNLP-Mixer-Base</td><td>90.00</td><td>98.33</td><td>1.3</td></tr>
  <tr><td>pNLP-Mixer-XL</td><td>91.03</td><td>98.4</td><td>5.0</td></tr>
  <tr><td>TS-Mixer-S</td><td>91.43</td><td>98.57</td><td>0.174</td></tr>
  <tr><td>TS-Mixer-B</td><td>91.99</td><td>98.69</td><td>0.429</td></tr>
  <tr><td>TS-Mixer-L</td><td>92.10</td><td>98.75</td><td>1.2</td></tr>
</table>
|table1|vlue|
|--|--|
| id | 1 |
| name | user |
