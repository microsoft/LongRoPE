import base64
import pickle
import transformers
from fairseq.data import Dictionary


IGNORE_IDX = -100

def get_tokenizer(tokenizer_name_or_path,
                  model_max_length=None,
                  default_bos_token="<s>",
                  default_eos_token="</s>",
                  default_pad_token="[PAD]",
                  default_unk_token="<unk>"):

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        model_max_length=model_max_length,
        padding_side="right",
        trust_remote_code=True,
        # use_fast=False,
    )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = default_pad_token
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = default_eos_token
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = default_bos_token
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = default_unk_token

    tokenizer.add_special_tokens(special_tokens_dict)
    if model_max_length:
        tokenizer.model_max_length = model_max_length
    return tokenizer


class HFDictionary(Dictionary):
    def __init__(
        self
    ):
        self.bos_word, self.unk_word, self.pad_word, self.eos_word = None, None, None, None
        self.symbols = []
        self.count = []
        self.indices = {}
        self.nspecial = len(self.symbols)

    def save(self, f):
        with open(f, "wb") as ff:
            pickle.dump([self.symbols, self.count, self.indices], ff)
    
    @classmethod
    def load(cls, f, bos=None, pad=None, eos=None, unk=None):
        d = cls()
        with open(f, "rb") as ff:
            d.symbols, d.count, d.indices = pickle.load(ff)

        assert bos in d.indices
        assert pad in d.indices
        assert eos in d.indices
        assert unk in d.indices
        d.set_special_tokens(bos, pad, eos, unk)
        return d

    def set_special_tokens(self, bos=None, pad=None, eos=None, unk=None):
        if bos:
            assert bos in self.indices
            self.bos_index = self.index(bos)
            self.bos_word = bos
        if pad:
            assert pad in self.indices
            self.pad_index = self.index(pad)
            self.pad_word = pad
        if eos:
            assert eos in self.indices
            self.eos_index = self.index(eos)
            self.eos_word = eos
        if unk:
            assert unk in self.indices
            self.unk_index = self.index(unk)
            self.unk_word = unk


def get_dict(llm_model_name_or_path):
    tokenizer = get_tokenizer(llm_model_name_or_path)
    fairseq_dict = HFDictionary()
    sorted_vocab = sorted(tokenizer.get_vocab().items(), key=lambda item: item[1])

    for token, index in sorted_vocab:

        # NOTE: This is a hack to ensure that
        while len(fairseq_dict) < index:
            fairseq_dict.add_symbol(f"<|cube_dummy_{len(fairseq_dict)}|>")

        if isinstance(token, bytes):
            fairseq_dict.add_symbol(base64.b64encode(token).decode("utf-8"))
        else:
            fairseq_dict.add_symbol(token)

    for token, index in tokenizer.get_vocab().items():
        if isinstance(token, bytes):
            assert fairseq_dict.index(base64.b64encode(token).decode("utf-8")) == index, f'token {token}, idx in hf {index}, idx in fs {fairseq_dict.index(base64.b64encode(token).decode("utf-8"))}'
        else:
            assert fairseq_dict.index(token) == index, f'token {token}, idx in hf {index}, idx in fs {fairseq_dict.index(token)}'

    fairseq_dict.set_special_tokens(bos=tokenizer.bos_token, pad=tokenizer.pad_token, eos=tokenizer.eos_token, unk=tokenizer.unk_token)
    return fairseq_dict
