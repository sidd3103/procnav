from procnav.cache import JsonCache, cache_key


def test_roundtrip_and_lazy_dir(tmp_path):
    d = tmp_path / "nested" / "cache"
    c = JsonCache(d)
    assert not d.exists()
    assert c.get("k") is None
    c.put("k", {"a": [1, 2]})
    assert (d / "k.json").exists()
    assert c.get("k") == {"a": [1, 2]}


def test_keys_differ_and_are_hex():
    a = cache_key("x", "y")
    assert a != cache_key("x", "z")
    assert a != cache_key("xy", "")
    assert len(a) == 64 and int(a, 16) >= 0
    assert a == cache_key("x", "y")


def test_corrupt_file_returns_none(tmp_path):
    c = JsonCache(tmp_path)
    (tmp_path / "bad.json").write_text('{"a": ')
    assert c.get("bad") is None


def test_put_leaves_no_temp_files(tmp_path):
    c = JsonCache(tmp_path)
    c.put("k", {"a": 1})
    c.put("k", {"a": 2})
    assert [p.name for p in tmp_path.iterdir()] == ["k.json"]
    assert c.get("k") == {"a": 2}
