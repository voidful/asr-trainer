# xls-r-fine-tuning

xls-r fine-tuning script modify from https://huggingface.co/blog/fine-tune-xlsr-wav2vec2

- fix large memory usage during eval metric calculation
- add cer and wer for evaluation

## example usage

`pip install -r requirements.txt`

`python train.py --common_voice_subset zh-TW --tokenize_config voidful/wav2vec2-large-xlsr-53-tw-gpt --xlsr_config facebook/wav2vec2-xls-r-300m`
