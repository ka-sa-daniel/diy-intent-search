from diy_search.models import Example, Size
from diy_search.vector import EmbeddingCache, VectorSearcher

# 単語ごとに決まった向きを持つ、APIを呼ばない偽の埋め込み
AXES = {"網戸": 0, "棚": 1, "傘": 2}


class FakeEmbedder:
    model = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        self.calls += 1
        out = []
        for t in texts:
            v = [0.01, 0.01, 0.01]
            for word, axis in AXES.items():
                if word in t:
                    v[axis] += 1.0
            out.append(v)
        return out


def make(id_: str, title: str) -> Example:
    return Example(
        id=id_, title=title, category="c", purpose="", actions=[], location="室内",
        size_cm=Size(), max_load_kg=None, constraints=[], tools=[], description="", materials=[],
    )


EXAMPLES = [make("A", "網戸の張り替え"), make("B", "壁付けの棚"), make("C", "傘立て")]


def test_ranks_by_cosine(tmp_path) -> None:
    s = VectorSearcher(EXAMPLES, FakeEmbedder(), cache_path=tmp_path / "c.json")
    assert [r.example_id for r in s.search("棚がほしい", k=2)][0] == "B"


def test_allowed_ids_filters_candidates(tmp_path) -> None:
    s = VectorSearcher(EXAMPLES, FakeEmbedder(), cache_path=tmp_path / "c.json")
    assert all(r.example_id != "B" for r in s.search("棚", allowed_ids={"A", "C"}))


def test_cache_avoids_recomputing_documents(tmp_path) -> None:
    path = tmp_path / "c.json"
    e = FakeEmbedder()
    VectorSearcher(EXAMPLES, e, cache_path=path)
    VectorSearcher(EXAMPLES, e, cache_path=path)
    assert e.calls == 1


def test_cache_key_changes_with_text() -> None:
    assert EmbeddingCache.key("m", "t", "a") != EmbeddingCache.key("m", "t", "b")
