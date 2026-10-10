"""One error type, one JSON envelope, and the Retry-After header that goes with it.

The envelope is OpenAI's (``{"error": {"message", "type", "code"}}``) plus
``retry_after_s`` when waiting is the right answer. A client that hedges (the
DisPatch voice add-on) reads only two things: the status and whether a
``Retry-After`` came with it -- a 503 with one means "the rig will be back, use
your local engine meanwhile"; anything else is a fault worth logging.
"""

from __future__ import annotations

from typing import Any


class SpeechError(Exception):
    status_code = 500
    code = "internal_error"
    error_type = "server_error"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        status_code: int | None = None,
        retry_after_s: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        if status_code is not None:
            self.status_code = status_code
        self.retry_after_s = retry_after_s
        self.details = details or {}

    def to_payload(self) -> dict[str, Any]:
        error: dict[str, Any] = {
            "message": self.message,
            "type": self.error_type,
            "code": self.code,
        }
        if self.retry_after_s is not None:
            error["retry_after_s"] = int(max(1, round(self.retry_after_s)))
        if self.details:
            error["details"] = self.details
        return {"error": error}

    def headers(self) -> dict[str, str]:
        if self.retry_after_s is None:
            return {}
        return {"Retry-After": str(int(max(1, round(self.retry_after_s))))}


class BadRequest(SpeechError):
    status_code = 400
    code = "invalid_request"
    error_type = "invalid_request_error"


class Unauthorized(SpeechError):
    status_code = 401
    code = "invalid_token"
    error_type = "invalid_request_error"


class Forbidden(SpeechError):
    status_code = 403
    code = "forbidden"
    error_type = "invalid_request_error"


class NotFound(SpeechError):
    status_code = 404
    code = "not_found"
    error_type = "invalid_request_error"


class TooLarge(SpeechError):
    status_code = 413
    code = "too_large"
    error_type = "invalid_request_error"


class Unavailable(SpeechError):
    """The GPU side cannot serve right now. Always carries a Retry-After."""

    status_code = 503
    code = "not_ready"
    error_type = "server_error"


class ConfigError(Exception):
    """A configuration the server refuses to start with (fail closed)."""
