import argparse
import sys

import asrp
from datasets import load_dataset, Audio
import torch
from dataclasses import dataclass
from typing import Dict, List, Union
from transformers import Wav2Vec2FeatureExtractor, EarlyStoppingCallback
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
        parser.add_argument("--common_voice_subset", type=str, required=True, default="zh-TW")
        parser.add_argument("--tokenize_config", type=str, required=True,
                            default="voidful/wav2vec2-large-xlsr-53-tw-gpt")
        parser.add_argument("--xlsr_config", type=str, required=True, default="facebook/wav2vec2-xls-r-300m")
        parser.add_argument("--batch", type=int, default=8)
        parser.add_argument("--grad_accum", type=int, default=2)
        input_arg, model_arg = parser.parse_known_args(args)
        input_arg = {k: v for k, v in vars(input_arg).items() if v is not None}
        other_arg = {k.replace("--", ""): v for k, v in zip(model_arg[:-1:2], model_arg[1::2])}
        return input_arg, other_arg

    def prepare_dataset(batch):
        audio = batch["audio"]

        # batched output is "un-batched"
        batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        batch["input_length"] = len(batch["input_values"])

        with processor.as_target_processor():
            batch["labels"] = processor(batch["sentence"]).input_ids
        return batch

    def compute_metrics(pred):
        pred_ids = pred.predictions
        pred_ids = [i[i != -100] for i in pred_ids]
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True, group_tokens=False)
        # we do not want to group tokens when computing the metrics
        label_ids = pred.label_ids
        label_ids = [i[i != -100] for i in label_ids]
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True, group_tokens=False)

        cer = asrp.cer(label_str, pred_str)
        wer = asrp.wer(label_str, pred_str)
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
            """
            Calling this function will disable the gradient computation for the feature extractor so that its parameter
            will not be updated during training.
            """
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
            r"""
            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_length)`, `optional`):
                Labels for connectionist temporal classification. Note that ``target_length`` has to be smaller or equal to
                the sequence length of the output logits. Indices are selected in ``[-100, 0, ..., config.vocab_size -
                1]``. All labels set to ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ...,
                config.vocab_size - 1]``.
            """

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
        """
        Data collator that will dynamically pad the inputs received.
        Args:
            processor (:class:`~transformers.Wav2Vec2Processor`)
                The processor used for proccessing the data.
            padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
                among:
                * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided.
                * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
                  different lengths).
        """

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
    common_voice_train = load_dataset("common_voice", input_arg['common_voice_subset'], split="train+validation")
    common_voice_test = load_dataset("common_voice", input_arg['common_voice_subset'], split="test")
    common_voice_train = common_voice_train.remove_columns(
        ["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
    common_voice_test = common_voice_test.remove_columns(
        ["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(input_arg['tokenize_config'])
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                                 do_normalize=True,
                                                 return_attention_mask=True)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    common_voice_train = common_voice_train.cast_column("audio", Audio(sampling_rate=16_000))
    common_voice_test = common_voice_test.cast_column("audio", Audio(sampling_rate=16_000))
    common_voice_train = common_voice_train.map(prepare_dataset, remove_columns=common_voice_train.column_names)
    common_voice_test = common_voice_test.map(prepare_dataset, remove_columns=common_voice_test.column_names)
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    model = Wav2Vec2ForCTC.from_pretrained(
        input_arg['xlsr_config'],
        attention_dropout=0.0,
        hidden_dropout=0.0,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.0,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )
    model.freeze_feature_extractor()
    repo_name = f"{input_arg['xlsr_config']}-{input_arg['common_voice_subset']}"
    training_args = TrainingArguments(
        output_dir=repo_name,
        group_by_length=True,
        per_device_train_batch_size=int(input_arg['batch']),
        per_device_eval_batch_size=int(input_arg['batch']),
        gradient_accumulation_steps=int(input_arg['grad_accum']),
        evaluation_strategy="steps",
        num_train_epochs=100,
        gradient_checkpointing=True,
        fp16=True,
        save_steps=400,
        eval_steps=400,
        logging_steps=400,
        learning_rate=3e-4,
        warmup_steps=500,
        save_total_limit=2,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=common_voice_train,
        eval_dataset=common_voice_test,
        tokenizer=processor.feature_extractor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    trainer.train()


if __name__ == "__main__":
    main()
