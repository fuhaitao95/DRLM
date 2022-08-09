# DRLM: A robust drug representation learning method and its applications

## 1 Introduction

High-quality and robust drug representations can broaden the understanding of pharmacology, and improve the modeling of multiple drug-related prediction tasks, which further facilitates drug development. DRLM is such a framework to learn drug representations with integrating gene expression profiles of drug-related cells and the therapeutic use information of drugs. And the learned drug representations by DRLM can be used for various drug-related downstream tasks, such as drug-disease association prediction, drug-drug interaction prediction and miRNA-drug resistance association prediction tasks.

## 2 Overview

Here we provide an implementation of DRLM with Python. The repository is organised as follows:

- `dataset/drugMiRNAData/` contains the necessary dataset files for predicting miRNA-drug resistance associations, which comes from the paper  of Niu et al.[<sup>1</sup>](#R1) The drug-disease association dataset and the drug-drug interaction dataset can obtained from the paper of Zhang et al.[<sup>2</sup>](#R2) and the paper of Liu et al.[<sup>3</sup>](#R3), respectively.
- `ourMethod/method/` contains the code scripts of our proposed method. Details are as follows:
  - `representationLearning.py` consists of three modules, namely, the stacked autoencoder, the iterative clustering module and the therapeutic use discriminator.
  -  `expCV.py` performs 5-fold cross-validation experiments for downstream tasks.
  - `experiments.py` puts all code scripts together and executes a full training run.

Below is an example command line for evaluating the model on the miRNA-drug resistance association prediction task: 

__Step 1__ (to learn the drug representations):

```shell
python experiments.py --expName representationLearning
```

__Step 2__ (to evaluate on the miRNA-drug resistance association prediction task):

```shell
python experiments.py --expName expCV --dataName drugMiRNA --clfName RF
```

We used the [DESC](https://eleozzr.github.io/desc/) algorithm during constructing our framework. We thank them a lot for their code sharing!

## 3 Requirements

python==3.7.1

keras==2.2.5

tensorflow==1.15.0

anndata==0.7.8

scanpy==1.7.2

sklearn==0.24.2

numpy==1.21.2

pandas==1.2.0

## 4 Citation

>@INPROCEEDINGS{Zhao2021robust,  
>
>author={Zhao, Cecheng and Huang, Ziyang and Wang, Hui and Fu, Haitao and Wang, Dong and Gao, Yingjie and Zhu, Haotian and Niu, Xiaohui and Zhang, Wen},  
>
>booktitle={2021 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)}, 
>
>title={A robust drug representation learning model for eliminating cell specificity in gene expression profile and its application}, 
>
>year={2021}, 
>
>pages={1191-1196}, 
>
>doi={10.1109/BIBM52615.2021.9669385}
>
>}

## 5 Contact

- Please feel free to contact us if you need any help: [fuhaitao@webmail.hzau.edu.cn](mailto:fuhaitao@webmail.hzau.edu.cn) OR [fuhaitao95@qq.com](mailto:fuhaitao95@qq.com)
- **Attention**: Only real name emails will be replied. Please provide as much detail as possible about the problem you are experiencing.
- **注意**：只回复实名电子邮件。请尽可能详细地描述您遇到的问题，可以附上截图等。

<div><a name="R1"></a>
    [1] Niu, Y., Song, C., Gong, Y., & Zhang, W. (2022). MiRNA-Drug Resistance Association Prediction Through the Attentive Multimodal Graph Convolutional Network. Frontiers in pharmacology, 12, 799108. https://doi.org/10.3389/fphar.2021.799108
</div>

<div><a name="R2"></a>
    [2] Zhang, W., Yue, X., Lin, W. et al. (2018). Predicting drug-disease associations by using similarity constrained matrix factorization. BMC Bioinformatics 19, 233. https://doi.org/10.1186/s12859-018-2220-4
</div>

<div><a name="R3"></a>
    [3] Liu, S., Zhang, Y., Cui, Y., Qiu, Y., Deng, Y., Zhang, Z. M., & Zhang, W. (2022). Enhancing drug-drug interaction prediction using deep attention neural networks. IEEE/ACM Transactions on Computational Biology and Bioinformatics.
</div>
