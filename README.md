# Domain Adaption in One-Shot Learning

Implementation of Domain Adaption in One-Shot Learning.

## Paper
You can temporarily find our paper at [here](https://nanqingd.github.io/files/ECML_2018_Camera_Ready_Final.pdf).
*The proceedings are coming out soon!*

## Citation
If you find DAOSL useful in your research, please consider to cite:

@inproceedings{dong2018domain, 
  title={Domain Adaption in One-Shot Learning}, 
  author={Dong, Nanqing and Eric P. Xing}, 
  booktitle={Joint European Conference on Machine Learning and Knowledge Discovery in Databases},
  pages={},
  year={2018}
}

## Setup
### Requirements
```
python 3.5
tensorflow 1.8
CUDA  9.0
cuDNN 7.0
```

### Installation
```
sh setup.sh
```

## Training
Train one-shot classifier for 5-way 1-shot learning.
```
python3 convert_data.py --data-name=omniglot
python3 convert_data.py --data-name=emnist --num-target-examples=20
python3 train_one_shot.py --exp-name=one_shot --source=omniglot --target=emnist_20 --num-ways=5
```

Train adversarial domain adaption (ADA) for 5-way 1-shot learning.
```
python3 convert_data.py --data-name=omniglot
python3 convert_data.py --data-name=emnist --num-target-examples=20
python3 train_ada.py --exp-name=ada --source=omniglot --target=emnist_20 --num-ways=5 --la=0.001
```

Train adversarial domain adaption (ADA) with reinforced sample selection (RSS) for 5-way 1-shot learning.
```
python3 convert_data.py --data-name=chars
python3 convert_data.py --data-name=sim
python3 convert_data.py --data-name=dis
python3 train_rss.py --exp-name=rss --num-ways=5 --la=0.001 --gamma=0.1
```