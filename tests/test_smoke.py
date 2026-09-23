import rpb


def test_version_is_string() -> None:
    assert isinstance(rpb.__version__, str)
