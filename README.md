# Adversarial Self-Attention

This repo is for the AAAI 2023 paper [Adversarial Self-Attention for Language Understanding](https://arxiv.org/abs/2206.12608).

<table>
  <tr>
    <td><div><center><img src="https://github.com/gingasan/adversarialSA/blob/main/figures/sa_map.png"
                          alt="sa"
                          style="zoom:70%;"/>
      <br>
      Self-Attention
      </center></div></td>
    <td><div><center><img src="https://github.com/gingasan/adversarialSA/blob/main/figures/asa_map.png"
                          alt="asa"
                          style="zoom:70%;"/>
      <br>
      Adversarial Self-Attention
      </center></div></td>
  </tr>
</table>



**Dependency**

* torch 1.9

* transformer 4.17



**Quick start**

Sentence classification

```bash
python run_sent_clas.py \
    --do_train \
    --do_eval \
    --task_name SST-2 \
    --learning_rate 2e-5 \
    --train_batch_size 32 \
    --do_lower_case \
    --model_type bert \
    --load_model_path bert-base-uncased \
    --output_dir sst_bert \
    --fp16
```

```txt
Epoch 0: global step = 2105 | train loss = 0.250 | eval score = 92.55 | eval loss = 0.211
Epoch 1: global step = 4210 | train loss = 0.114 | eval score = 93.00 | eval loss = 0.202
Epoch 2: global step = 6315 | train loss = 0.073 | eval score = 93.46 | eval loss = 0.223
```

Testing:

```bash
python run_sent_clas.py \
    --do_test \
    --task_name SST-2 \
    --learning_rate 2e-5 \
    --eval_batch_size 128 \
    --do_lower_case \
    --model_type bert \
    --load_model_path bert-base-uncased \
    --output_dir sst_bert \
    --test_model_file sst_bert/2_pytorch_model.bin
```



Multiple choices

```bash
python run_multi_cho.py \
    --do_train \
    --do_eval \
    --task_name DREAM \
    --eval_on test \
    --num_train_epochs 6 \
    --learning_rate 2e-5 \
    --train_batch_size 16 \
    --model_type roberta \
    --load_model_path roberta-base \
    --output_dir dream_roberta \
    --fp16
```

```txt
Epoch 0: global step = 242 | loss = 1.066 | eval score = 56.41 | eval loss = 0.908
Epoch 1: global step = 484 | loss = 0.825 | eval score = 67.13 | eval loss = 0.749
Epoch 2: global step = 726 | loss = 0.540 | eval score = 68.76 | eval loss = 0.731
Epoch 3: global step = 968 | loss = 0.329 | eval score = 69.54 | eval loss = 0.867
Epoch 4: global step = 1210 | loss = 0.221 | eval score = 69.70 | eval loss = 0.966
Epoch 5: global step = 1452 | loss = 0.167 | eval score = 69.23 | eval loss = 1.037
```

