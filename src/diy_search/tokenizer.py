"""SudachiPyによる単語分割。表記ゆれを正規化形でそろえる（ネジ/ねじ -> 螺子）。"""

from functools import lru_cache

from sudachipy import Dictionary, SplitMode

# 検索に使う品詞。助詞・助動詞・記号などは捨てる
_KEEP_POS = {"名詞", "動詞", "形容詞", "形状詞", "副詞", "接尾辞", "接頭辞"}
# よく出るが意味の薄い語
_STOPWORDS = {"する", "ある", "いる", "なる", "できる", "ため", "こと", "もの", "よう", "ところ"}


@lru_cache(maxsize=1)
def _tokenizer():
    return Dictionary(dict="core").create()


def tokenize(text: str) -> list[str]:
    """短い単位（A）で分ける。「突っ張り棒」->「突っ張り」「棒」となり、部分一致しやすくする。"""
    out = []
    for m in _tokenizer().tokenize(text, SplitMode.A):
        if m.part_of_speech()[0] not in _KEEP_POS:
            continue
        word = m.normalized_form()
        if word in _STOPWORDS:
            continue
        out.append(word)
    return out
