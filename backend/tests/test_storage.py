"""Prompt 011-014 测试：Storage 抽象、LocalStorage、安全检查、StorageService。"""

from __future__ import annotations

import pytest
from app.core.exceptions import StorageException
from app.storage.local import LocalStorage
from app.storage.security import ensure_inside_root, validate_key


@pytest.fixture()
def local(tmp_path) -> LocalStorage:
    return LocalStorage(root=tmp_path / "root")


# ---------- security.validate_key ----------
@pytest.mark.parametrize(
    "bad_key",
    [
        "../etc/passwd",
        "a/../../b",
        "..\\windows\\evil",
        "/etc/passwd",
        "C:/Windows/system32",
        "D:\\x",
        "a//b",
        "a/./b",
        "con",
        "aux/table.csv",
        "bad<name>.csv",
        'file"quote.csv',
        "",
        "   ",
    ],
)
def test_validate_key_rejects(bad_key):
    with pytest.raises(StorageException):
        validate_key(bad_key)


@pytest.mark.parametrize(
    "good_key,expected",
    [("raw/a.csv", "raw/a.csv"), ("a\\b\\c.csv", "a/b/c.csv"), ("x.CSV", "x.CSV")],
)
def test_validate_key_accepts(good_key, expected):
    assert validate_key(good_key) == expected


def test_ensure_inside_root_rejects_symlink_escape(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "root"
    root.mkdir()
    link = root / "leak"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink requires privileges on this system")
    with pytest.raises(StorageException, match="escapes"):
        ensure_inside_root(root, link / "file.txt")


# ---------- LocalStorage ----------
def test_save_read_roundtrip(local):
    meta = local.save("raw/a.csv", b"a,b\n1,2\n")
    assert meta.key == "raw/a.csv"
    assert meta.size == 8
    assert local.read("raw/a.csv") == b"a,b\n1,2\n"
    assert local.exists("raw/a.csv")


def test_save_creates_parent_dirs(local):
    local.save("datasets/d1/v000001.parquet", b"data")
    assert local.exists("datasets/d1/v000001.parquet")


def test_read_not_found(local):
    with pytest.raises(StorageException) as exc:
        local.read("raw/missing.csv")
    assert exc.value.code == "NOT_FOUND"


def test_stat_and_metadata(local):
    local.save("raw/b.csv", b"xyz")
    meta = local.stat("raw/b.csv")
    assert meta.size == 3
    assert meta.modified_at is not None


def test_delete(local):
    local.save("raw/c.csv", b"1")
    local.delete("raw/c.csv")
    assert not local.exists("raw/c.csv")
    with pytest.raises(StorageException):
        local.delete("raw/c.csv")


def test_list_by_prefix(local):
    local.save("raw/1.csv", b"1")
    local.save("raw/2.csv", b"2")
    local.save("datasets/x.parquet", b"3")
    keys = {m.key for m in local.list("raw/")}
    assert keys == {"raw/1.csv", "raw/2.csv"}


def test_local_rejects_traversal(local):
    with pytest.raises(StorageException):
        local.save("../evil.txt", b"x")
    with pytest.raises(StorageException):
        local.read("../../etc/passwd")


# ---------- StorageService ----------
def test_storage_service_facade(storage):
    storage.save("raw/f.csv", b"hello")
    assert storage.exists("raw/f.csv")
    assert storage.read("raw/f.csv") == b"hello"
    assert storage.metadata("raw/f.csv").size == 5
    assert [m.key for m in storage.list("raw/")] == ["raw/f.csv"]
    storage.delete("raw/f.csv")
    assert not storage.exists("raw/f.csv")
