# Machine Learning Localization

This repository is the official implementation of: 

[Mitigating loss of variance in ensemble data assimilation: machine learning-based and distance-free localization for better covariance estimation](https://arxiv.org/pdf/2506.13362)

## Directories

- **data**: data files to test the ML-localization.
- **ml_localization**: ML-localization module folder.

## Requirements

To install requirements:

```setup
 $ pip install -r requirements.txt
```

## Usage

To run ML-localization (help):

```runh1
 $ python -m ml_localization -h
```
or
```runh2
 $ python ./run.py -h
```

To run ML-localization on **data** folder:

```setup
 $ python -m ml_localization -m 20 -d 1530 -M data/M.bin -D data/D.bin -Ms data/Msuper.bin -Ds data/Dsuper.bin -R data/R.bin -l data/ml_localization.log
```
or
```setup
 $ python ./run.py -m 20 -d 1530 -M data/M.bin -D data/D.bin -Ms data/Msuper.bin -Ds data/Dsuper.bin -R data/R.bin -l data/ml_localization.log
```