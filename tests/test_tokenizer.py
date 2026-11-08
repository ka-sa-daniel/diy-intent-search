from diy_search.tokenizer import tokenize


def test_normalizes_spelling_variants() -> None:
    assert tokenize("ネジ") == tokenize("ねじ")


def test_drops_particles_and_symbols() -> None:
    tokens = tokenize("棚を付けたい。")
    assert "を" not in tokens and "。" not in tokens
    assert "棚" in tokens and "付ける" in tokens


def test_splits_compound_words() -> None:
    assert "突っ張り" in tokenize("突っ張り棒")
