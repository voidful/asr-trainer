# xls-r-fine-tuning

xls-r fine-tuning script modify from https://huggingface.co/blog/fine-tune-xlsr-wav2vec2

- fix large memory usage during eval metric calculation
- add cer and wer for evaluation

## install requirement

`pip install -r requirements.txt`

## example usage - common voice
`python -m torch.distributed.launch --nproc_per_node=2 train.py --common_voice_subset zh-TW --tokenize_config voidful/wav2vec2-large-xlsr-53-tw-gpt --xlsr_config facebook/wav2vec2-xls-r-300m --batch 10 --group_by_length --max_input_length_in_sec 20`

## example usage - custom set

### custom data format `data.csv`:

```csv
path,text
/xxx/2.wav,被你拒絕而記仇
/xxx/4.wav,電影界的人
/xxx/7.wav,其實我最近在想
```

`python -m torch.distributed.launch --nproc_per_node=2 --custom_set ./data.csv --tokenize_config voidful/wav2vec2-large-xlsr-53-tw-gpt --xlsr_config facebook/wav2vec2-xls-r-300m --batch 3 --grad_accum 15 --max_input_length_in_sec 15 --eval_step 10000`

## sweep usage

`python -m wandb sweep ./sweep_xxx.yaml`   
`python -m wandb agent xxxxxxxxxxxxxx`
