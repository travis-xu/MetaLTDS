# MetaLTDS
Code for our paper [Balanced Meta Learning and Diverse Sampling for Lifelong Task-Oriented Dialogue Systems](https://ojs.aaai.org/index.php/AAAI/article/view/26621/26393) (AAAI2023). ![MetaLTDS](/assets/MetaLTDS.png)

## Installation

```
pip install -r requirements.txt
cd data
bash download.sh
```

## Train

```
bash train_tmab.sh
```

## Test

```
bash test.sh
```

## Note

This code is based on [andreamad8/ToDCL: Continual Learning for Task-Oriented Dialogue Systems](https://github.com/andreamad8/ToDCL).

## Citation

```
@inproceedings{xu2023balanced,
  title={Balanced meta learning and diverse sampling for lifelong task-oriented dialogue systems},
  author={Xu, Qiancheng and Yang, Min and Xu, Ruifeng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={37},
  number={11},
  pages={13843--13852},
  year={2023}
}
```
