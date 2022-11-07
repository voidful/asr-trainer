import argparse
import sys

from pyctcdecode import build_ctcdecoder
from pyctcdecode.alphabet import BLANK_TOKEN_PTN, UNK_TOKEN, UNK_TOKEN_PTN
from transformers import AutoProcessor
from transformers import Wav2Vec2ProcessorWithLM


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--lm", type=str, required=True)
    input_arg, model_arg = parser.parse_known_args(args)
    input_arg = {k: v for k, v in vars(input_arg).items() if v is not None}
    other_arg = {k.replace("--", ""): v for k, v in zip(model_arg[:-1:2], model_arg[1::2])}
    return input_arg, other_arg


def main(arg=None):
    input_arg, other_arg = parse_args(sys.argv[1:]) if arg is None else parse_args(arg)

    processor = AutoProcessor.from_pretrained(input_arg['model'])
    vocab_dict = processor.tokenizer.get_vocab()
    sorted_vocab_dict = {k: v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}
    alphabet = list(sorted_vocab_dict.keys())

    decoder = build_ctcdecoder(
        labels=alphabet,
        kenlm_model_path=input_arg['lm'],
    )

    for i, token in enumerate(alphabet):
        if BLANK_TOKEN_PTN.match(token):
            alphabet[i] = ""
        if token == processor.tokenizer.word_delimiter_token:
            alphabet[i] = " "
        if UNK_TOKEN_PTN.match(token):
            alphabet[i] = UNK_TOKEN

    decoder._alphabet._labels = alphabet
    processor_with_lm = Wav2Vec2ProcessorWithLM(
        feature_extractor=processor.feature_extractor,
        tokenizer=processor.tokenizer,
        decoder=decoder
    )

    processor_with_lm.save_pretrained(input_arg['model'])


if __name__ == "__main__":
    main()
