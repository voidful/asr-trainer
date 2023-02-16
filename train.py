import inspect
import random
import sys

import nlp2
from datasets import load_dataset, Audio
from transformers import Seq2SeqTrainer
from transformers import Trainer
from transformers import TrainingArguments, Seq2SeqTrainingArguments
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
from transformers import WhisperFeatureExtractor
from transformers import WhisperForConditionalGeneration
from transformers import WhisperProcessor

from module.args import parse_args
from module.data_processing import encode_dataset, DataCollatorCTCWithPadding, prepare_dataset_hf, \
    prepare_dataset_custom, DataCollatorSpeechSeq2SeqWithPadding
from module.g2p import G2P
from module.metric import cer_cal, wer_cal
from module.model import Wav2Vec2ForCTC
from module.utility import FreezingCallback


def main(arg=None):
    input_arg, other_arg = parse_args(sys.argv[1:]) if arg is None else parse_args(arg)
    print("input_arg", input_arg)
    repo_name = f"{input_arg['model_config']}-{input_arg['custom_set_train'] if 'custom_set_train' in input_arg else input_arg['train_subset']}"
    repo_name = repo_name.replace("/", "_")

    if 'openai/whisper' in input_arg['model_config']:
        feature_extractor = WhisperFeatureExtractor.from_pretrained(input_arg['model_config'])
        processor = WhisperProcessor.from_pretrained(input_arg['model_config'], task="transcribe")
        processor.save_pretrained(repo_name)
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []
        audio_feature_key = inspect.getfullargspec(model.forward).args[1]
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, audio_feature_key=audio_feature_key)
    else:
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(input_arg['tokenize_config'],
                                                         use_auth_token=input_arg['use_auth_token'])
        feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                                     do_normalize=True,
                                                     return_attention_mask=True)
        processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        processor.save_pretrained(repo_name)
        model = Wav2Vec2ForCTC.from_pretrained(
            input_arg['model_config'],
            activation_dropout=input_arg.get('activation_dropout', 0.01),
            attention_dropout=input_arg.get('attention_dropout', 0.01),
            feat_proj_dropout=input_arg.get('feat_proj_dropout', 0.01),
            feat_quantizer_dropout=input_arg.get('feat_quantizer_dropout', 0.01),
            final_dropout=input_arg.get('final_dropout', 0.01),
            hidden_dropout=input_arg.get('hidden_dropout', 0.01),
            layerdrop=0.0,
            ctc_loss_reduction="mean",
            pad_token_id=processor.tokenizer.pad_token_id,
            vocab_size=len(processor.tokenizer),
            use_auth_token=input_arg['use_auth_token'],
            ignore_mismatched_sizes=True
        )
        audio_feature_key = inspect.getfullargspec(model.forward).args[1]
        data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True,
                                                   audio_feature_key=audio_feature_key)

    # data set
    if 'custom_set_train' in input_arg:
        dataset = load_dataset('csv', data_files=input_arg['custom_set_train'], cache_dir=input_arg['cache_dir'])
        dataset = dataset.filter(lambda e: nlp2.is_file_exist(e['path']))
        if 'custom_set_test' in input_arg:
            dataset_test = load_dataset('csv', data_files=input_arg['custom_set_test'],
                                        cache_dir=input_arg['cache_dir'])
            dataset_test = dataset_test.filter(lambda e: nlp2.is_file_exist(e['path']))
            data_test = dataset_test['train']
        else:
            dataset = dataset['train'].train_test_split(test_size=0.1)
            data_test = dataset['test']

        data_train = dataset['train']
        data_train = data_train.map(prepare_dataset_custom, num_proc=input_arg["num_proc"],
                                    fn_kwargs={'audio_feature_key': audio_feature_key})
        data_test = data_test.map(prepare_dataset_custom, num_proc=input_arg["num_proc"],
                                  fn_kwargs={'audio_feature_key': audio_feature_key})
    elif 'train_set' in input_arg:
        data_train = load_dataset(input_arg['train_set'], input_arg['train_subset'],
                                  split=input_arg['train_split'], use_auth_token=input_arg['use_auth_token'])
        data_test = load_dataset(input_arg.get('test_set', input_arg['train_set']),
                                 input_arg['test_subset'] if 'test_subset' in input_arg else input_arg['train_subset'],
                                 split=input_arg['test_split'],
                                 use_auth_token=input_arg['use_auth_token'])
        try:
            data_train = data_train.remove_columns(
                ["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
            data_test = data_test.remove_columns(
                ["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
        except:
            pass

        data_train = data_train.cast_column("audio", Audio(sampling_rate=16_000))
        data_test = data_test.cast_column("audio", Audio(sampling_rate=16_000))

        data_train = data_train.map(prepare_dataset_hf,
                                    fn_kwargs={'processor': processor, 'audio_feature_key': audio_feature_key},
                                    remove_columns=data_train.column_names)
        data_test = data_test.map(prepare_dataset_hf,
                                  fn_kwargs={'processor': processor, 'audio_feature_key': audio_feature_key},
                                  remove_columns=data_test.column_names)
    print("init dataset")
    print("data train", data_train)
    print("data test", data_test)

    if not input_arg.get('load_cache', False):
        print("before filtering audio length")
        print("data train", data_train)
        print("data test", data_test)
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
        print("after filtering audio length")
        print("data train", data_train)
        print("data test", data_test)

        print("before filtering label length")
        print("data train", data_train)
        print("data test", data_test)
        data_train = data_train.filter(
            lambda x: x is not None and 0 < len(x),
            input_columns=["labels"])
        data_test = data_test.filter(
            lambda x: x is not None and 0 < len(x),
            input_columns=["labels"])
        print("after filtering label length")
        print("data train", data_train)
        print("data test", data_test)

        print("before encoding dataset")
        print("data train", data_train)
        print("data test", data_test)
        phonemize = input_arg.get('phoneme', False)
        if phonemize:
            if 'g2p' in phonemize:
                backend = G2P(delimiter=tokenizer.word_delimiter_token)
                separator = None
            else:
                from phonemizer.backend import EspeakBackend
                from phonemizer.separator import Separator
                separator = Separator(phone="", word=tokenizer.word_delimiter_token, syllable="")
                backend = EspeakBackend(language="en-us", language_switch="remove-flags")
            if not input_arg.get('only_eval', False):
                data_train = data_train.map(encode_dataset, fn_kwargs={'processor': processor,
                                                                       'phonemize': phonemize,
                                                                       'separator': separator,
                                                                       'backend': backend})
            data_test = data_test.map(encode_dataset,
                                      fn_kwargs={'processor': processor, 'phonemize': phonemize,
                                                 'separator': separator, 'backend': backend})
        else:
            if not input_arg.get('only_eval', False):
                data_train = data_train.map(encode_dataset,
                                            fn_kwargs={'processor': processor})
            data_test = data_test.map(encode_dataset,
                                      fn_kwargs={'processor': processor})
        print("after encoding dataset")
        print("data train", data_train)
        print("data test", data_test)

        data_train.save_to_disk(f"{repo_name}-train.data")
        data_test.save_to_disk(f"{repo_name}-test.data")
    else:
        data_train.load_from_disk(f"{repo_name}-train.data")
        data_test.load_from_disk(f"{repo_name}-test.data")

    print("finalize dataset")
    print("data train", data_train)
    print("data test", data_test)
    print("train labels", data_train[0]['labels'])
    print("test labels", data_test[0]['labels'])

    if input_arg.get('sweep_split_shard', False):
        shuffled_dataset = data_train.shuffle(seed=42)
        data_train = shuffled_dataset.shard(num_shards=input_arg.get('sweep_split_shard'), index=0)
        data_train = data_train.shard(num_shards=input_arg.get('sweep_split_shard'), index=0)
        data_test = data_train

    if 'openai/whisper' in input_arg['model_config']:
        trainer_class = Seq2SeqTrainer
        trainer_aug_class = Seq2SeqTrainingArguments
    else:
        trainer_class = Trainer
        trainer_aug_class = TrainingArguments
        model.gradient_checkpointing_enable()

    training_args = trainer_aug_class(
        output_dir=input_arg.get("output_dir", repo_name),
        length_column_name="lengths",
        group_by_length=input_arg["group_by_length"],
        per_device_train_batch_size=int(input_arg['batch']),
        per_device_eval_batch_size=int(input_arg['batch']),
        gradient_accumulation_steps=int(input_arg['grad_accum']),
        eval_accumulation_steps=int(input_arg['grad_accum']),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_steps=input_arg.get('eval_steps', 400),
        eval_steps=input_arg.get('eval_steps', 400),
        ddp_find_unused_parameters=True,
        resume_from_checkpoint=input_arg.get("checkpoint", False),
        overwrite_output_dir=input_arg.get("overwrite_output_dir", False),
        load_best_model_at_end=True,
        greater_is_better=False,
        metric_for_best_model='cer',
        num_train_epochs=input_arg.get('epoch', 60),
        fp16=True,
        logging_steps=input_arg.get('logging_steps', 10),
        learning_rate=input_arg.get('learning_rate', 2.34e-4),
        warmup_steps=input_arg.get('warmup_steps', 100),
        save_total_limit=input_arg.get('save_total_limit', 5),
        push_to_hub=False,
        report_to="all"
    )
    if 'openai/whisper' in input_arg['model_config']:
        training_args.predict_with_generate = True
        training_args.generation_max_length = 225

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
        pred_result = [[l, p, cer_cal([l], [p])] for l, p in zip(label_str, pred_str)]
        nlp2.write_csv(pred_result, 'pred.csv')
        # print 10 predict result randomly for debug
        random.shuffle(pred_result)
        print("pred_result")
        print("=================================")
        for i in range(10):
            print(pred_result[i])
        print("=================================")
        return {"cer": cer, "wer": wer}

    trainer = trainer_class(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=data_train,
        eval_dataset=data_test,
        tokenizer=processor.feature_extractor,
    )

    if not input_arg.get('only_eval', False):
        freezing_callback = FreezingCallback(trainer, model, input_arg.get('unfreeze_warmup_steps', 1000))
        trainer.add_callback(freezing_callback)
        trainer.train(input_arg.get("checkpoint", None))
    else:
        trainer.evaluate()


if __name__ == "__main__":
    main()
