from sfp.trainers.common import _deep_merge


def test_deep_merge_preserves_nested_base_keys():
    base = {"a": 1, "nested": {"x": 1, "y": 2}}
    override = {"nested": {"y": 10}}
    merged = _deep_merge(base, override)

    assert merged["a"] == 1
    assert merged["nested"]["x"] == 1
    assert merged["nested"]["y"] == 10
