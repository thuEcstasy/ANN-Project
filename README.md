# ANN Project: DiffuSeq using Jittor Framework
# Yihe Wang, Ningyuan Zhang, Zhaofeng Sun
# 2024.12.19

## Train DiffuSeq using PyTorch
1. install the required packages
```
cd DiffuSeq-main
```

in `requirements.txt`:
```
bert_score
blobfile
nltk
numpy==1.23.1
packaging
psutil
PyYAML
setuptools
spacy

torch==1.9.0+cu111 # using `pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html`

torchmetrics
tqdm
transformers==4.22.2
wandb
datasets
```

2. train
Change `nproc_per_node` to `1` since we only have one GPU on the server

```
cd scripts
python -m torch.distributed.launch --nproc_per_node=1 --master_port=12233 --use_env run_train.py --diff_steps 2000 --lr 0.0001 --learning_steps 2000 --save_interval 1000 --seed 102 --noise_schedule sqrt --hidden_dim 128 --bsz 2048 --dataset qqp --data_dir /home/aiuser/ANN/DiffuSeq-main/datasets/QQP --vocab bert --seq_len 128 --schedule_sampler lossaware --notes qqp
```

3. hyperparameter tuning
![alt text](image.png)

In `scripts/run_train.py`, `hidden_t_dim`,`hidden_dim`, `diff_steps`, transformer model hyperparameters (i.e. #layers, #attention_heads) can be tuned to reach the best performance.

Modify the hyperparameters when starting training.

4. DiffuB
Use the file in `scripts/DiffuB-DiffuSeq`, and in `diffuseq/utils/DiffuB-DiffuSeq`, with all others unchanged.
