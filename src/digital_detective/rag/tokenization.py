"""Shared lexical tokenization for the controlled retrieval conditions."""

from __future__ import annotations

import re

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(_TOKEN_PATTERN.findall(text.lower()))
