import pickle
import transformers
from fairseq.data import Dictionary

def get_tokenizer(llm_model_name_or_path, max_seq_len):
    DEFAULT_PAD_TOKEN = "[PAD]"
    DEFAULT_EOS_TOKEN = "</s>"
    DEFAULT_BOS_TOKEN = "<s>"
    DEFAULT_UNK_TOKEN = "<unk>"

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        llm_model_name_or_path,
        model_max_length=max_seq_len,
        padding_side="right",
        # use_fast=False,
    )
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    tokenizer.add_special_tokens(special_tokens_dict)
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
        with open(f, 'wb') as ff:
            pickle.dump([self.symbols, self.count, self.indices], ff)
    
    @classmethod
    def load(cls, f, bos=None, pad=None, eos=None, unk=None):
        d = cls()
        with open(f, 'rb') as ff:
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


def get_dict(llm_model_name_or_path, max_seq_len):
    tokenizer = get_tokenizer(llm_model_name_or_path, max_seq_len)
    fairseq_dict = HFDictionary()
    sorted_vocab = sorted(tokenizer.get_vocab().items(), key=lambda item: item[1]) 
    for token, index in sorted_vocab:
        fairseq_dict.add_symbol(token)

    for token, index in tokenizer.get_vocab().items():
        assert fairseq_dict.index(token) == index, f'token {token}, idx in hf {index}, idx in fs {fairseq_dict.index(token)}'

    fairseq_dict.set_special_tokens(bos=tokenizer.bos_token, pad=tokenizer.pad_token, eos=tokenizer.eos_token, unk=tokenizer.unk_token)
    return fairseq_dict
