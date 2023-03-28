<h2 align="center">
Adap-$\tau$: Adaptively Modulating Embedding Magnitude for Recommendation
</h2>

<div align="center">

[![](https://img.shields.io/badge/paper-pink?style=plastic&logo=GitBook)](https://arxiv.org/pdf/2302.04775.pdf)
[![](https://img.shields.io/badge/-github-green?style=plastic&logo=github)](https://github.com/junkangwu/Adap_tau) 
<!-- [![](https://img.shields.io/badge/-youtube-red?style=plastic&logo=airplayvideo)](https://www.youtube.com/watch?v=9d0eXaO_kOw) 
[![](https://img.shields.io/badge/-slides-grey?style=plastic&logo=adobe)](https://cs.stanford.edu/~shirwu/slides/dir-iclr22.pdf)  -->
</div>

This is the PyTorch implementation for our WWW 2023 paper. 
> Jiawei Chen, Junkang Wu, Jiancan Wu, Sheng Zhou, Xuezhi Cao, Xiangnan He. 2023. Adap-$\tau$: Adaptively Modulating Embedding Magnitude for Recommendation [arxiv link](https://arxiv.org/pdf/2302.04775.pdf)

## Dependencies
- pytorch==1.11.0
- numpy==1.21.5
- scipy==1.7.3
- torch-scatter==2.0.9

## Training model:
- mkdir log
- cd bash
### MF
#### yelp2018
```
# Adap_tau_0
bash Adap_tau_novel.sh  yelp2018 1e-4 1e-3 3 1024 2048 drop 1.0 1.0 uniform_gpu 0 100 cosine mf weight_v0
# Adap_tau
bash Adap_tau_novel.sh  yelp2018 1e-4 1e-3 3 1024 2048 drop 1.0 1.0 uniform_gpu 0 100 cosine mf weight_mean
```
#### amazon-book
```
# Adap_tau_0
bash Adap_tau_novel.sh amazon-book 1e-3 1e-7 3 1024 2048 nopdrop 1.0 1.0 uniform_gpu 0 100 cosine mf weight_v0
# Adap_tau
bash Adap_tau_novel.sh amazon-book 1e-3 1e-7 3 1024 2048 nopdrop 1.0 1.0 uniform_gpu 0 100 cosine mf weight_mean
```
#### gowalla
```
# Adap_tau_0
bash Adap_tau_novel.sh gowalla 1e-4 1e-9 3 1024 2048 drop 0.9 0.25 uniform_gpu 0 100 cosine mf weight_v0
# Adap_tau
bash Adap_tau_novel.sh gowalla 1e-4 1e-9 3 1024 2048 drop 0.9 0.25 uniform_gpu 0 100 cosine mf weight_ratio
```

### LightGCN
#### yelp2018
```
# Adap_tau_0
bash Adap_tau_novel.sh  yelp2018 1e-3 1e-1 3 1024 2048 drop 1.0 1.0 no_sample 0 100 nocosine lgn weight_v0
# Adap_tau
bash Adap_tau_novel.sh  yelp2018 1e-3 1e-1 3 1024 2048 drop 1.0 1.5 no_sample 0 100 nocosine lgn weight_mean
```
#### amazon-book
```
# Adap_tau_0
bash Adap_tau_novel.sh amazon-book 1e-4 1e-1 3 1024 2048 nopdrop 1.0 1.0 no_sample 0 100 nocosine lgn weight_v0
# Adap_tau
bash Adap_tau_novel.sh amazon-book 1e-4 1e-1 3 1024 2048 nopdrop 1.0 1.0 no_sample 0 100 nocosine lgn weight_mean
```

#### gowalla
```
# Adap_tau_0
bash Adap_tau_novel.sh gowalla 1e-3 1e-5 3 1024 2048 nopdrop 0.8 0.6 no_sample 0 100 nocosine lgn weight_v0
# Adap_tau
bash Adap_tau_novel.sh gowalla 1e-3 1e-5 3 1024 2048 nopdrop 0.8 0.6 no_sample 0 100 nocosine lgn weight_mean
```
111
The [training log](https://github.com/junkangwu/Adap_tau/tree/master/outputs) is also provided. The results fluctuate slightly under different running environment.

For any clarification, comments, or suggestions please create an issue or contact me (jkwu0909@mail.ustc.edu.cn).