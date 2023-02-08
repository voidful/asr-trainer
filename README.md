# one script for xls-r/xlsr/whisper fine-tuning

script modify from https://huggingface.co/blog/fine-tune-xlsr-wav2vec2

- fix large memory usage during eval metric calculation
- add cer and wer for evaluation

## install requirement

`pip install -r requirements.txt`

## example usage - common voice
```
python -m torch.distributed.launch --nproc_per_node=2 \
train.py \
--train_subset zh-TW \
--train_split train \
--test_split validation \
--tokenize_config voidful/wav2vec2-large-xlsr-53-tw-gpt \
--model_config facebook/wav2vec2-xls-r-300m \
--batch 10 \
--group_by_length \
--max_input_length_in_sec 20
```

```
python -m torch.distributed.launch --nproc_per_node=2 \
train.py \
--train_subset zh-TW \
--model_config openai/whisper-base \
--batch 10 \
--group_by_length \
--max_input_length_in_sec 20
```

## example usage - custom set

### custom data format `data.csv`:

```csv
path,text
/xxx/2.wav,被你拒絕而記仇
/xxx/4.wav,電影界的人
/xxx/7.wav,其實我最近在想
```

```
python -m torch.distributed.launch --nproc_per_node=2 \
--custom_set ./data.csv \
--tokenize_config voidful/wav2vec2-large-xlsr-53-tw-gpt \
--model_config facebook/wav2vec2-xls-r-300m \
--batch 3 \
--grad_accum 15 \
--max_input_length_in_sec 15 \
--eval_step 10000
```

```shell
python -m torch.distributed.launch --nproc_per_node=2 \
train.py --tokenize_config facebook/hubert-large-ls960-ft \
--model_config ntu-spml/distilhubert \
--group_by_length \
--train_set librispeech_asr \
--train_subset all \
--train_split train.clean.100+train.clean.360+train.other.500 \
--test_split test.other \
--learning_rate 0.0003 \
--batch 30 \
--logging_steps 10 \
--eval_steps 60 \
--epoch 150 \
--use_auth_token True \
--output_dir ./model_sweep \
--overwrite_output_dir
```

Train whisper on custom dataset
```shell
python train.py --tokenize_config openai/whisper-base \
--model_config openai/whisper-base \
--group_by_length \
--custom_set_train cm_train_unified.csv \
--custom_set_test cm_dev_unified.csv \
--output_dir ./whisper-custom \
--overwrite_output_dir
```

## sweep usage

`python -m wandb sweep ./sweep_xxx.yaml`   
`python -m wandb agent xxxxxxxxxxxxxx`

