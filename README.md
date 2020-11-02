# TLAT-NMT
Source code for the EMNLP 2020 long paper &lt;Token-level Adaptive Training for Neural Machine Translation>.


## Related code

Implemented based on [Fairseq-py](https://github.com/pytorch/fairseq), an open-source toolkit released by Facebook which was implemented strictly referring to [Vaswani et al. (2017)](https://arxiv.org/pdf/1706.03762.pdf).

## Requirements
This system has been tested in the following environment.
+ OS: Ubuntu 16.04.1 LTS 64 bits
+ Python version \>=3.7
+ Pytorch version \>=1.0

## Replicate the En-De results

Download [the preprocessed WMT'16 EN-DE data provided by Google](https://drive.google.com/uc?export=download&id=0B_bZck-ksdkpM25jRUN2X2UxMm8) and preprocess it following the [instrucition](https://github.com/ictnlp/awesome-transformer).

Pretrain the model for about 30 epochs.

```
bash train.ende.sh
```

Continual train the model based on the last checkpoint with the adaptive weights for about 15 epochs.

```
bash train.ende.ft.sh
```

Inference
```
$ python generate.py wmt16_en_de_bpe32k --path $SMODEL \
    --gen-subset test --beam 4 --batch-size 128 \
    --remove-bpe --lenpen 0.6 > pred.de \
# because fairseq's output is unordered, we need to recover its order
$ grep ^H pred.de | cut -f1,3- | cut -c3- | sort -k1n | cut -f2- > pred.de
```

## Citation
```
@inproceedings{gu2020token,
  title={Token-level Adaptive Training for Neural Machine Translation},
  author={Gu, Shuhao and Zhang, Jinchao and Meng, Fandong and Feng, Yang and Xie, Wanying and Zhou, Jie and Yu, Dong},
  journal={EMNLP2020},
  year={2020}
}
```
