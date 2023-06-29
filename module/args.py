import argparse


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--custom_set_train", type=str)
    parser.add_argument("--custom_set_test", type=str)
    parser.add_argument("--cache_dir", type=str, default='./.cache')
    parser.add_argument("--train_set", type=str)
    parser.add_argument("--train_subset", type=str)
    parser.add_argument("--train_split", type=str)
    parser.add_argument("--test_set", type=str)
    parser.add_argument("--test_subset", type=str)
    parser.add_argument("--test_split", type=str)
    parser.add_argument("--tokenize_config", type=str, default="openai/whisper-small")
    parser.add_argument("--model_config", type=str, default="openai/whisper-small")
    parser.add_argument("--sweep_split_shard", type=int)
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--logging_steps", type=int)
    parser.add_argument("--eval_steps", type=int)
    parser.add_argument("--phoneme", choices=["espeak", "g2p"])
    parser.add_argument("--unit", choices=['hubert_layer9_code500', 'hubert_layer9_code500_norm_beam'])
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--group_by_length", action="store_true")
    parser.add_argument("--use_auth_token", action="store_true")
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
    parser.add_argument("--warmup_steps", type=int)
    parser.add_argument("--save_total_limit", type=int)
    parser.add_argument("--only_eval", action="store_true")
    input_arg, model_arg = parser.parse_known_args(args)
    input_arg = {k: v for k, v in vars(input_arg).items() if v is not None}
    other_arg = {k.replace("--", ""): v for k, v in zip(model_arg[:-1:2], model_arg[1::2])}
    return input_arg, other_arg


def parse_args_create_tokenizer(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--custom_set_train", type=str)
    parser.add_argument("--custom_set_test", type=str)
    parser.add_argument("--train_set", type=str)
    parser.add_argument("--train_subset", type=str)
    parser.add_argument("--train_split", type=str)
    parser.add_argument("--test_set", type=str)
    parser.add_argument("--test_subset", type=str)
    parser.add_argument("--test_split", type=str)
    parser.add_argument("--vocab_list", type=str)
    parser.add_argument("--output_dir", type=str)
    input_arg, model_arg = parser.parse_known_args(args)
    input_arg = {k: v for k, v in vars(input_arg).items() if v is not None}
    other_arg = {k.replace("--", ""): v for k, v in zip(model_arg[:-1:2], model_arg[1::2])}
    return input_arg, other_arg
