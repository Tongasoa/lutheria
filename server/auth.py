"""Authentification du client micro (token comparé en temps constant)."""

import hmac


def is_valid_mic_token(provided: str | None, expected: str) -> bool:
    if not provided:
        return False
    return hmac.compare_digest(provided, expected)
