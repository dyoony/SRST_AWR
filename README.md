# SRST_AWR
This repository contains the code for ICCV 2023 paper "Enhancing Adversarial Robustness in Low-Label Regime via Adaptively Weighted Regularizationand Knowledge Distillation" by Dongyoon Yang, Insung Kong and Yongdai Kim.



# Train

`python main.py'


# Evaluation

The trained models can be evaluated by running eval.py which contains the standard accuracy and robust accuracies against PGD and AutoAttack.

`python eval.py --datadir {data_dir} --model_dir {model_dir} --swa --model resnet18 --attack_method autoattack`

# Citation


```
@inproceedings{
    dongyoon2023enhancing,
    title={Enhancing Adversarial Robustness in Low-Label Regime via Adaptively Weighted Regularizationand Knowledge Distillation},
    author={Dongyoon Yang, Insung Kong and Yongdai Kim},
    booktitle={International Conference on Computer Vision},
    year={2023},
    url={https://arxiv.org/abs/2308.04061}
}
```
