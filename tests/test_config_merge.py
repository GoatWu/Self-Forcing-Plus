from sfp.trainers.common import _deep_merge


def test_deep_merge_adds_new_keys():
    base = {"a": 1}
    override = {"b": 2}
    merged = _deep_merge(base, override)

    assert merged["a"] == 1
    assert merged["b"] == 2
