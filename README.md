# SRST_AWR
This repository contains the code for ICCV 2023 paper "Enhancing Adversarial Robustness in Low-Label Regime via Adaptively Weighted Regularizationand Knowledge Distillation" by Dongyoon Yang, Insung Kong and Yongdai Kim.

If you have some questions, please leave comments.

# Train

## Train Teacher

`python train_teacher.py --dataset {dataset} --model {model} --depth {depth} --widen_factor {widen_factor} --num_labels {num_labels} --algo fixmatch --lamb 1 --eta 0.95`

## Train Model

`python main.py --dataset cifar10 --model wideresnet --depth 28 --widen_factor 5 --num_labels 4000 --algo srst-awr --perturb_loss kl --teacher fixmatch --tau 1.2 --smooth 0.2 --lamb 20 --gamma 4 --beta 0.5 --lr 0.05 --swa`

`python main.py --dataset cifar100 --model wideresnet --depth 28 --widen_factor 8 --num_labels 4000 --algo srst-awr --perturb_loss ce --teacher fixmatch --tau 1.0 --smooth 0.2 --lamb 20 --gamma 4 --beta 0.5 --lr 0.05 --swa`

`python main.py --dataset stl10 --model wideresnet --depth 28 --widen_factor 5 --num_labels 1000 --algo srst-awr --perturb_loss ce --teacher fixmatch --tau 1.0 --smooth 0.2 --lamb 8 --gamma 4 --beta 0.5 --lr 0.05 --swa`

# Evaluation

The trained models can be evaluated by running eval.py which contains the standard accuracy and robust accuracies against PGD and AutoAttack.

`python eval.py --datadir {data_dir} --model_dir {model_dir} --swa --model {model} --depth {depth} --widen_factor {widen_factor} --attack_method autoattack`

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
