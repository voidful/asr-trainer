import json
import os.path
import string
import sys

import nlp2
from transformers import Wav2Vec2CTCTokenizer

from module.args import parse_args_create_tokenizer

chars_to_ignore = "､／｛`％＼/„⋯·『｢*？〝)°￼）>\u3000＆{＜＊^=‛]}!@〃〾℃；.〈﹐《〖─﹁%〔+」，¥’”〙︰﹂：〛】．｠〟﹑_＇｜｝\"｀•＠~〕［\\－&–｡‟$‘＃（〰！[＞〗‧＂,〜〞;｟﹏\'＋-｣＝(＾“「|～#﹖〚﹣。?＿<…:》＄【］、〿﹔〉—〘⠀』"


def main(arg=None):
    input_arg, other_arg = parse_args_create_tokenizer(sys.argv[1:]) if arg is None else parse_args_create_tokenizer(
        arg)
    print("input_arg", input_arg)

    def token_check(tok):
        if tok not in char_lexicon and \
                tok not in list(string.ascii_lowercase) and \
                tok not in list(string.digits) and \
                tok not in chars_to_ignore:
            return True
        return False

    char_lexicon = []
    for l in nlp2.read_files_into_lines(input_arg['vocab_list']):
        char = l.strip()
        if token_check(char):
            char_lexicon.append(char)
    char_lexicon = list(string.ascii_lowercase) + list(string.digits) + char_lexicon
    special_token = ["[UNK]", "[PAD]", "|"]
    char_lexicon += special_token
    vocab_dict = {v: k for k, v in enumerate(set(char_lexicon))}
    nlp2.get_dir_with_notexist_create(input_arg['output_dir'])
    with open(os.path.join(input_arg['output_dir'], "./vocab.json"), 'w', encoding='utf8') as vocab_file:
        json.dump(vocab_dict, vocab_file, ensure_ascii=False)

    tokenizer = Wav2Vec2CTCTokenizer(os.path.join(input_arg['output_dir'], "./vocab.json"),
                                     unk_token="[UNK]",
                                     pad_token="[PAD]",
                                     word_delimiter_token="|")

    tokenizer.save_pretrained(input_arg['output_dir'])


if __name__ == "__main__":
    main()
