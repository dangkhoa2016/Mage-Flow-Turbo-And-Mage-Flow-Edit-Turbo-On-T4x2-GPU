"""Dependency-free public API-token value authority.

This module is the single source of truth for what counts as a valid public
API token *value*. Every boundary that accepts a token - the coordinator auth
middleware, the persisted token store, and the shell startup validation -
must call the same function so none of them can silently disagree:

- the server rejects a token that is too short;
- the token store must also reject (fail closed) a persisted token that the
  server would reject;
- the shell launcher must not start a coordinator with a token the server
  would refuse.

The module intentionally imports nothing outside the standard library: a token
value does not need FastAPI or any framework dependency.
"""

from __future__ import annotations

MIN_PUBLIC_TOKEN_LENGTH = 32


class TokenContractError(ValueError):
    """Raised when a token value violates the shared public contract."""


def validate_public_token_value(token: object) -> str:
    """Validate and return an API token value under the public contract.

    Required contract (fail closed):

    - the value must be a ``str``;
    - its length must be at least ``MIN_PUBLIC_TOKEN_LENGTH``;
    - it must not be whitespace-only;
    - it must not contain a LF, CR, or NUL character;
    - it must be free of leading/trailing whitespace so a value read from a file
      can never be ambiguous with the exact value the server compares.

    Returns the unchanged value on success, raising ``TokenContractError`` on
    the first violation.
    """
    if not isinstance(token, str):
        raise TokenContractError("API token must be a string")
    if token != token.strip():
        raise TokenContractError("API token must not contain leading/trailing whitespace")
    if len(token) < MIN_PUBLIC_TOKEN_LENGTH:
        raise TokenContractError(f"API token must be at least {MIN_PUBLIC_TOKEN_LENGTH} characters")
    if not token.strip():
        raise TokenContractError("API token must not be whitespace-only")
    if "\n" in token:
        raise TokenContractError("API token must not contain line feed characters")
    if "\r" in token:
        raise TokenContractError("API token must not contain carriage return characters")
    if "\x00" in token:
        raise TokenContractError("API token must not contain NUL bytes")
    return token
