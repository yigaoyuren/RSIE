# RSIE
## Data
The dataset used in the paper can be found here：https://drive.google.com/drive/my-drive?hl=zh-tw

Change the address of your dataset in configs/base.py

Before training please create a folder called checkpoints under the root directory
## Dependencies
GeForce RTX 3080 

conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
## Train
``` 
python3 main.py --exp zhang_exp --config zhang_best
```
## Test
```
python3 eval.py --exp zhang_exp
```
