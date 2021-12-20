import argparse
import os
import sys

import editdistance as ed
import torchaudio
from datasets import load_dataset, Audio
import torch
from dataclasses import dataclass
from typing import Dict, List, Union
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2PreTrainedModel, Wav2Vec2Model
import torch.nn as nn
from transformers import TrainingArguments
from transformers import Trainer

_HIDDEN_STATES_START_POSITION = 2
from transformers.modeling_outputs import CausalLMOutput


def main(arg=None):
    def parse_args(args):
        parser = argparse.ArgumentParser()
        parser.add_argument("--custom_set", type=str)
        parser.add_argument("--common_voice_subset", type=str)
        parser.add_argument("--tokenize_config", type=str,
                            default="voidful/wav2vec2-large-xlsr-53-tw-gpt")
        parser.add_argument("--xlsr_config", type=str, default="facebook/wav2vec2-xls-r-1b")
        parser.add_argument("--sweep_split_shard", type=int)
        parser.add_argument("--num_train_epochs", type=int)
        parser.add_argument("--batch", type=int, default=8)
        parser.add_argument("--logging_steps", type=int)
        parser.add_argument("--eval_steps", type=int)
        parser.add_argument("--output_dir", type=str)
        parser.add_argument("--overwrite_output_dir", action="store_true")
        parser.add_argument("--group_by_length", action="store_true")
        parser.add_argument("--grad_accum", type=int, default=2)
        parser.add_argument("--num_proc", type=int, default=10)
        parser.add_argument("--max_input_length_in_sec", type=int)
        parser.add_argument("--activation_dropout", type=float)
        parser.add_argument("--attention_dropout", type=float)
        parser.add_argument("--feat_proj_dropout", type=float)
        parser.add_argument("--feat_quantizer_dropout", type=float)
        parser.add_argument("--final_dropout", type=float)
        parser.add_argument("--hidden_dropout", type=float)
        parser.add_argument("--learning_rate", type=float)
        input_arg, model_arg = parser.parse_known_args(args)
        input_arg = {k: v for k, v in vars(input_arg).items() if v is not None}
        other_arg = {k.replace("--", ""): v for k, v in zip(model_arg[:-1:2], model_arg[1::2])}
        return input_arg, other_arg

    def prepare_dataset_cov(batch):
        audio = batch["audio"]

        # batched output is "un-batched"
        batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        batch["lengths"] = len(batch["input_values"])

        with processor.as_target_processor():
            batch["labels"] = processor(batch["sentence"]).input_ids
        return batch

    def prepare_dataset_custom(batch):
        path = batch["path"]
        speech, sampling_rate = torchaudio.load(path)
        if sampling_rate != '16_000' or sampling_rate != '16000':
            resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16_000)
            batch["input_values"] = resampler.forward(speech.squeeze(0)).numpy()
        else:
            batch["speech"] = speech.squeeze(0).numpy()
        batch["lengths"] = len(batch["input_values"])

        with processor.as_target_processor():
            batch["labels"] = tokenizer(batch["text"]).input_ids
        return batch

    def cer_cal(groundtruth, hypothesis):
        err = 0
        tot = 0
        for p, t in zip(hypothesis, groundtruth):
            err += float(ed.eval(p.lower(), t.lower()))
            tot += len(t)
        return err / tot

    def wer_cal(groundtruth, hypothesis):
        err = 0
        tot = 0
        for p, t in zip(hypothesis, groundtruth):
            p = p.lower().split(' ')
            t = t.lower().split(' ')
            err += float(ed.eval(p, t))
            tot += len(t)
        return err / tot

    def compute_metrics(pred):
        pred_ids = pred.predictions
        pred_ids = [i[i != -100] for i in pred_ids]
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True, group_tokens=False)
        # we do not want to group tokens when computing the metrics
        label_ids = pred.label_ids
        label_ids = [i[i != -100] for i in label_ids]
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True, group_tokens=False)

        cer = cer_cal(label_str, pred_str)
        wer = wer_cal(label_str, pred_str)
        return {"cer": cer, "wer": wer}

    class Wav2Vec2ForCTC(Wav2Vec2PreTrainedModel):
        def __init__(self, config):
            super().__init__(config)

            self.wav2vec2 = Wav2Vec2Model(config)
            self.dropout = nn.Dropout(config.final_dropout)

            if config.vocab_size is None:
                raise ValueError(
                    f"You are trying to instantiate {self.__class__} with a configuration that "
                    "does not define the vocabulary size of the language model head. Please "
                    "instantiate the model as follows: `Wav2Vec2ForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                    "or define `vocab_size` of your model's configuration."
                )
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

            self.init_weights()

        def freeze_feature_extractor(self):
            self.wav2vec2.feature_extractor._freeze_parameters()

        def forward(
                self,
                input_values,
                attention_mask=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                labels=None,
        ):
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
            outputs = self.wav2vec2(
                input_values,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = outputs[0]
            hidden_states = self.dropout(hidden_states)

            logits = self.lm_head(hidden_states)

            loss = None
            if labels is not None:

                if labels.max() >= self.config.vocab_size:
                    raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

                # retrieve loss input_lengths from attention_mask
                attention_mask = (
                    attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
                )
                input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

                # assuming that padded tokens are filled with -100
                # when not being attended to
                labels_mask = labels >= 0
                target_lengths = labels_mask.sum(-1)
                flattened_targets = labels.masked_select(labels_mask)

                # ctc_loss doesn't support fp16
                log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
                logits = torch.argmax(logits, -1)
                with torch.backends.cudnn.flags(enabled=False):
                    loss = nn.functional.ctc_loss(
                        log_probs,
                        flattened_targets,
                        input_lengths,
                        target_lengths,
                        blank=self.config.pad_token_id,
                        reduction=self.config.ctc_loss_reduction,
                        zero_infinity=self.config.ctc_zero_infinity,
                    )

            if not return_dict:
                output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
                return ((loss,) + output) if loss is not None else output

            return CausalLMOutput(
                loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
            )

    @dataclass
    class DataCollatorCTCWithPadding:
        processor: Wav2Vec2Processor
        padding: Union[bool, str] = True

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            # split inputs and labels since they have to be of different lenghts and need
            # different padding methods
            input_features = [{"input_values": feature["input_values"]} for feature in features]
            label_features = [{"input_ids": feature["labels"]} for feature in features]

            batch = self.processor.pad(
                input_features,
                padding=self.padding,
                return_tensors="pt",
            )
            with self.processor.as_target_processor():
                labels_batch = self.processor.pad(
                    label_features,
                    padding=self.padding,
                    return_tensors="pt",
                )

            # replace padding with -100 to ignore loss correctly
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

            batch["labels"] = labels
            return batch

    input_arg, other_arg = parse_args(sys.argv[1:]) if arg is None else parse_args(arg)
    print("input_arg", input_arg)
    repo_name = f"{input_arg['xlsr_config']}-{input_arg['custom_set'] if 'custom_set' in input_arg else input_arg['common_voice_subset']}"

    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(input_arg['tokenize_config'])
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                                 do_normalize=True,
                                                 return_attention_mask=True)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    processor.save_pretrained(repo_name)
    # data set
    if 'custom_set' in input_arg:
        cache_file_train = f"{input_arg['custom_set']}_hf_train.data"
        cache_file_test = f"{input_arg['custom_set']}_hf_test.data"
        if os.path.isdir(cache_file_train) and os.path.isdir(cache_file_test):
            data_train = load_dataset('csv', data_files=input_arg['custom_set'])['train']
            data_test = load_dataset('csv', data_files=input_arg['custom_set'])['train']
            data_train = data_train.load_from_disk(cache_file_train)
            data_test = data_test.load_from_disk(cache_file_test)
        else:
            dataset = load_dataset('csv', data_files=input_arg['custom_set'], cache_dir='./.cache')
            dataset = dataset['train']
            dataset = dataset.train_test_split(test_size=0.1)
            data_train = dataset['train']
            data_test = dataset['test']
            data_train = data_train.map(prepare_dataset_custom, keep_in_memory=False, num_proc=input_arg["num_proc"])
            data_test = data_test.map(prepare_dataset_custom, keep_in_memory=False, num_proc=input_arg["num_proc"])
            data_train.save_to_disk(cache_file_train)
            data_test.save_to_disk(cache_file_test)
    elif 'common_voice_subset' in input_arg:
        data_train = load_dataset("common_voice", input_arg['common_voice_subset'], split="train+validation")
        data_test = load_dataset("common_voice", input_arg['common_voice_subset'], split="test")
        data_train = data_train.remove_columns(
            ["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
        data_test = data_test.remove_columns(
            ["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
        data_train = data_train.cast_column("audio", Audio(sampling_rate=16_000))
        data_test = data_test.cast_column("audio", Audio(sampling_rate=16_000))
        data_train = data_train.map(prepare_dataset_cov, remove_columns=data_train.column_names)
        data_test = data_test.map(prepare_dataset_cov, remove_columns=data_test.column_names)

    if input_arg.get('max_input_length_in_sec', None):
        max_input_length_in_sec = input_arg['max_input_length_in_sec']
        min_input_length_in_sec = 1
        data_train = data_train.filter(
            lambda
                x: min_input_length_in_sec * processor.feature_extractor.sampling_rate < x < max_input_length_in_sec * processor.feature_extractor.sampling_rate,
            input_columns=["lengths"])
        data_test = data_test.filter(
            lambda
                x: min_input_length_in_sec * processor.feature_extractor.sampling_rate < x < max_input_length_in_sec * processor.feature_extractor.sampling_rate,
            input_columns=["lengths"])

    data_train = data_train.filter(
        lambda x: 0 < len(x),
        input_columns=["labels"])
    data_test = data_test.filter(
        lambda x: 0 < len(x),
        input_columns=["labels"])

    if input_arg.get('sweep_split_shard', False):
        shuffled_dataset = data_train.shuffle(seed=42)
        data_train = shuffled_dataset.shard(num_shards=input_arg.get('sweep_split_shard'), index=0)
        data_train = data_train.shard(num_shards=input_arg.get('sweep_split_shard'), index=0)
        data_test = data_train

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    model = Wav2Vec2ForCTC.from_pretrained(
        input_arg['xlsr_config'],
        activation_dropout=input_arg.get('activation_dropout', 0.055),
        attention_dropout=input_arg.get('attention_dropout', 0.094),
        feat_proj_dropout=input_arg.get('feat_proj_dropout', 0.1),
        feat_quantizer_dropout=input_arg.get('feat_quantizer_dropout', 0.04),
        final_dropout=input_arg.get('final_dropout', 0.1),
        hidden_dropout=input_arg.get('hidden_dropout', 0.047),
        layerdrop=0.0,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )
    model.freeze_feature_extractor()
    training_args = TrainingArguments(
        output_dir=input_arg.get("output_dir", repo_name),
        length_column_name="lengths",
        group_by_length=input_arg["group_by_length"],
        per_device_train_batch_size=int(input_arg['batch']),
        per_device_eval_batch_size=int(input_arg['batch']),
        gradient_accumulation_steps=int(input_arg['grad_accum']),
        evaluation_strategy="steps",
        overwrite_output_dir=input_arg.get("overwrite_output_dir", False),
        load_best_model_at_end=True,
        num_train_epochs=input_arg.get('num_train_epochs', 60),
        gradient_checkpointing=True,
        fp16=True,
        save_steps=input_arg.get('eval_steps', 400),
        eval_steps=input_arg.get('eval_steps', 400),
        logging_steps=input_arg.get('logging_steps', 200),
        learning_rate=input_arg.get('learning_rate', 2.34e-4),
        warmup_steps=500,
        save_total_limit=2,
        push_to_hub=False,
    )
    model.gradient_checkpointing_enable()
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=data_train,
        eval_dataset=data_test,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()


if __name__ == "__main__":
    main()
