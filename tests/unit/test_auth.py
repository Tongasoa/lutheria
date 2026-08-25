"""Tests de la vérification du token producteur."""

from server.auth import is_valid_mic_token


def test_valid_token():
    assert is_valid_mic_token("abc", "abc") is True


def test_invalid_token():
    assert is_valid_mic_token("abc", "xyz") is False


def test_missing_token():
    assert is_valid_mic_token(None, "abc") is False


def test_empty_token():
    assert is_valid_mic_token("", "abc") is False
