# CasCIFF

![](https://img.shields.io/badge/python-3.7-green)
![](https://img.shields.io/badge/pytorch-1.10-green)
![](https://img.shields.io/badge/cudatoolkit-11.3.1-green)
![](https://img.shields.io/badge/cudnn-6.0-green)
 
This repo provides a reference implementation of **CasCIFF** as described in the paper:
> [CasCIFF: A Cross-Domain Information Fusion Framework Tailored for Cascade Prediction in Social Networks](https://arxiv.org/abs/2308.04961)  
> Hongjun Zhu, Member, IEEE, Shun Yuan, Xin Liu, Kuo Chen, Chaolong Jia, and Ying Qian  
> This paper will be released soon

## Basic Usage

### Requirements

The code was tested with `python 3.7`, `pytorch 1.10`, `cudatoolkit 11.3.1`, and `cudnn 6.0`. Install the dependencies via [Anaconda](https://www.anaconda.com/):

```shell
# create virtual environment
conda create --name casciff python=3.7 cudatoolkit=11.3.1 cudnn=6.0

# activate environment
conda activate casciff

# install other requirements
pip install -r requirements.txt
```

### Run the code
```shell
cd ./casCIFF

# generate information cascades
python ./preprocess/gene_cas.py 

# generate global graph embeddings
python ./preprocess/gene_global_emb.py 

# preprocess cascades data for training
python ./preprocess/preprocess_graph_signal_time.py

# preprocess global graph embeddings for training
python ./preprocess/preprocess_global_emb.py

# run CasCIFF model
python CasCIFF_train_shuffle.py
```
More running options are described in the codes.

## Evaluate
In addition, we also provide already trained models for you to reproduce the experimental effects in the paper.

```shell
# run evaluate CasCIFF model
python CasCIFF_metric.py
```

## Datasets


Our's datasets from [CasFlow](https://github.com/Xovee/casflow).

Thanks to [Xovee Xu](https://www.xoveexu.com/) for providing the dataset. Datasets download link: [Google Drive](https://drive.google.com/file/d/1o4KAZs19fl4Qa5LUtdnmNy57gHa15AF-/view?usp=sharing) or [Baidu Drive (password: `1msd`)](https://pan.baidu.com/s/1tWcEefxoRHj002F0s9BCTQ). 

The datasets we used in the paper are come from:

- [Twitter](http://carl.cs.indiana.edu/data/#virality2013) (Weng *et al.*, [Virality Prediction and Community Structure in Social Network](https://www.nature.com/articles/srep02522), Scientific Report, 2013).
- [Weibo](https://github.com/CaoQi92/DeepHawkes) (Cao *et al.*, [DeepHawkes: Bridging the Gap between 
Prediction and Understanding of Information Cascades](https://dl.acm.org/doi/10.1145/3132847.3132973), CIKM, 2017). You can also download Weibo dataset [here](https://drive.google.com/file/d/1fgkLeFRYQDQOKPujsmn61sGbJt6PaERF/view?usp=sharing) in Google Drive.  
- [APS](https://journals.aps.org/datasets) (Released by *American Physical Society*, obtained at Jan 17, 2019). 


## Contact

For any questions please open an issue or drop an email to: `zhuhj@cqupt.edu.cn, s211231071@stu.cqupt.edu.cn`
