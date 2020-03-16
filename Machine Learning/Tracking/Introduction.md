# Tracking

[CVPR 2019 论文汇总](<http://bbs.cvmart.net/topics/302/cvpr2019paper#4>)

近年的研究中图像特征提取部分基本以深度学习为主，但可考虑融合激光的几何特征。以下的分类基于论文提出相似度度量方法是采用传统的相关滤波还是采用深度学习方法。当特征提取和相似度度量部分均采用深度学习时，跟踪问题成为一个可以端到端训练的方法。

深度学习的研究着力点有几：

- 提出新的网络框架/改进已有网络框架
- Finetuning已有的网络，着力点可以使hard sample mining, data argument, 改进l损失函数等
- 将多个方法融合解决新的问题

## Correlation Filter



## CNN

### Unsupervised Tracking

Google在2019年ICML上的论文证明适用于各种场景的无监督学习并不可行，因此对于无监督方法首先要关注的是其专注于什么场景下的问题。

**Unsupervised Deep Tracking**

本文通过比较forward和backward tracking的一致性来实现无监督的目标跟踪。

论文提出的方法对目标的局部进行跟踪而非对完整目标的跟踪。

> 平衡样本都是在一个mini-batch中进行的



### Semi-supervised Tracking

该分支是近年研究的主流。基于模型是否需要在线训练可以分为online methods以及offline methods.

### Supervised Tracking

## GCN