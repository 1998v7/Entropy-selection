# Entropy-based selection for Learning with Noisy Labels

Official PyTorch Implementation of paper "A Joint Training Framework for Learning with Noisy Labels". 

> Paper "A Joint Training Framework for Learning with Noisy Labels" is accepted to **SCIENCE CHINA Information Sciences 2023**.

> 论文 “面向标签噪声的联合训练框架” 被 **中国科学-信息科学 2023** 接收.

# Training

For CIFAR-10, `warm_up = 10`,`model = resnet18`

For CIFAR-100, `warm_up = 30`,`model = resnet34`

### Run

```
python main.py --dataset cifar10 --model resnet18 --batch_size 32 --lr 0.02 --warm_up 10 \
                --num_epochs 100 --noise_mode instance --r 0.2 --k 2 --T 0.2 --gpuid 0
```

> Note that the code refers to DivideMix (ICLR 2021) and Self-Filtering (ECCV 2022). 
