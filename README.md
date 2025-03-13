# RSIE
## Title
Compact Structural Feature Enhancement for Unsupervised Anomaly Detection in Chest Radiographs
## Data
The dataset used in the paper can be found here：

ZhangLab：https://drive.google.com/file/d/1hOUIoPQlFcnY8f3ZdgOK7By7nRiTvNkX/view?usp=drive_link

Chexpert：https://drive.google.com/file/d/1NBwSfCI526ofT1IhFc-P3BvMqj-l1PMA/view?usp=drive_link

COVIDx：https://drive.google.com/file/d/1wM9nkMQPW2DJ7DDkfaRu-qmQYBtpr7_-/view?usp=drive_link

Change the address of your dataset in configs/base.py

Before training please create a folder called checkpoints under the root directory
## Pretrained Models
ZhangLab：https://drive.google.com/drive/folders/1RXahcep9ylwYyt2qZzQ-S8j5DzcNmfl7?usp=drive_link

Chexpert：(https://drive.google.com/file/d/1JGeK3Pp6A2B78ij48SXp6u-7kq9uRW8n/view?usp=sharing)

COVIDx：https://drive.google.com/drive/folders/1YiQADOY2YQqREOUhVmHD5GX2kDXOCnv_?usp=drive_link
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

Test data ：[https://docs.google.com/spreadsheets/d/1cmJyJhBN-yKZnucqWgHrEXRZ7Z-h9nqy/edit?usp=drive_link&ouid=105636907703743304488&rtpof=true&sd=true](https://docs.google.com/spreadsheets/d/1cmJyJhBN-yKZnucqWgHrEXRZ7Z-h9nqy/edit?usp=sharing&ouid=105636907703743304488&rtpof=true&sd=true)
## Citation
```
@article{ye2025rsie,
  title={Compact Structural Feature Enhancement for Unsupervised Anomaly Detection in Chest Radiographs},
  author={Jixun Ye,Wanhui Gao,Yun Wu and Ge Jiao},
  journal={The Visual Computer},
  year={2025}
}
```
