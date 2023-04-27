python main.py --dataset cifar10 --noise_mode sym --r 0.2  --penal_coeff 0.3 --T 3 --threshold 0.3 --main_type base --gpuid 0
python main.py --dataset cifar10 --noise_mode sym --r 0.5  --penal_coeff 0.3 --T 3 --threshold 0.3 --main_type base --gpuid 0
python main.py --dataset cifar10 --noise_mode pair --r 0.4  --penal_coeff 0.3 --T 3 --threshold 0.3 --main_type base --gpuid 0
python main.py --dataset cifar10 --noise_mode instance --r 0.2  --penal_coeff 0.3 --T 3 --threshold 0.3 --main_type base --gpuid 0
python main.py --dataset cifar10 --noise_mode instance --r 0.4  --penal_coeff 0.3 --T 3 --threshold 0.3 --main_type base --gpuid 0


python main.py --dataset cifar100 --noise_mode sym --r 0.2  --penal_coeff 0.3 --T 3 --threshold 0.3 --main_type base --gpuid 0
python main.py --dataset cifar100 --noise_mode sym --r 0.5  --penal_coeff 0.3 --T 3 --threshold 0.3 --main_type base --gpuid 0
python main.py --dataset cifar100 --noise_mode pair --r 0.4  --penal_coeff 0.3 --T 3 --threshold 0.3 --main_type base --gpuid 0
python main.py --dataset cifar100 --noise_mode instance --r 0.2  --penal_coeff 0.3 --T 3 --threshold 0.3 --main_type base --gpuid 0
python main.py --dataset cifar100 --noise_mode instance --r 0.4  --penal_coeff 0.3 --T 3 --threshold 0.3 --main_type base --gpuid 0