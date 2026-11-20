"""Groq LLM client: one chat function with a rate limiter and retries.

The model is set directly below. Groq retires models often, so if calls start failing
check console.groq.com/docs/deprecations and change MODEL.
"""
import logging
import os
import random
import threading
import time
from collections import deque

from dotenv import load_dotenv
from openai import APITimeoutError, OpenAI, RateLimitError

import json

from pydantic import BaseModel, ValidationError
from typing import Type, TypeVar

T = TypeVar("T", bound=BaseModel)

load_dotenv()  
logger = logging.getLogger(__name__)

# ---- settings ----
MODEL = "openai/gpt-oss-20b"
REASONING_EFFORT = "low"  # gpt-oss thinks before answering; "low" keeps replies fast
TEMPERATURE = 0.2
MAX_TOKENS = 4029  
TIMEOUT_SECONDS = 30
MAX_RETRIES = 4
CALLS_PER_MINUTE = 20  
BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_SYSTEM = "You are a helpful assistant."


class RateLimiter:
    """Sliding window: allows at most `max_calls` per `period` seconds.

    It makes us wait before a call instead of getting a 429 error back, which matters
    when many calls run in a loop (batch questions, evaluation).
    """

    def __init__(self, max_calls: int, period_seconds: float):
        self.max_calls = max_calls
        self.period = period_seconds
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self):
        while True:
            with self._lock:
                now = time.monotonic()
                while self._calls and now - self._calls[0] >= self.period:
                    self._calls.popleft()
                if len(self._calls) < self.max_calls:
                    self._calls.append(now)
                    return
                wait = self.period - (now - self._calls[0])
            logger.debug(f"[LLM] rate limiter waiting {wait:.2f}s")
            time.sleep(wait)  # sleep outside the lock so other threads are not blocked


limiter = RateLimiter(CALLS_PER_MINUTE, 60)
_client: OpenAI | None = None


def get_client() -> OpenAI:
    """OpenAI-compatible client pointed at Groq. Created once and reused."""
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set (put it in .env)")
        # max_retries=0: we do our own retries below, with backoff
        _client = OpenAI(api_key=api_key, base_url=BASE_URL, timeout=TIMEOUT_SECONDS, max_retries=0)
    return _client


def extract_retry_after(exc: RateLimitError) -> float | None:
    """The wait time the server asks for, if the 429 response includes one."""
    try:
        value = exc.response.headers.get("retry-after")
        return float(value) if value is not None else None
    except Exception:
        return None


def backoff_wait(attempt: int, exc: RateLimitError, base: float = 2.0, cap: float = 60.0) -> float:
    """Use the server's retry-after if given; otherwise capped exponential backoff plus jitter."""
    retry_after = extract_retry_after(exc)
    if retry_after is not None:
        return retry_after + random.uniform(0, 0.5)
    wait = min(cap, base * (2 ** (attempt - 1)))
    return wait + random.uniform(0, wait * 0.25)



def call_llm_raw(
    prompt: str,
    system: str = DEFAULT_SYSTEM,
    temperature: float = TEMPERATURE,
) -> str:
    """
    Call Groq LLM and return a plain text response.

    Use for freeform outputs where JSON wrapping is unnecessary.
    """
    client = get_client()
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            limiter.acquire()

            logger.debug(
                f"[LLM] groq/{MODEL} | "
                f"raw text call | attempt={attempt}"
            )

            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=MAX_TOKENS,
                reasoning_effort=REASONING_EFFORT,
            )

            raw = response.choices[0].message.content

            if not raw:
                raise ValueError("[LLM] Empty response received")

            raw = raw.strip()

            logger.debug(
                f"[LLM] raw success | chars={len(raw)}"
            )

            return raw

        except RateLimitError as e:
            last_error = e
            wait = backoff_wait(attempt, e)

            logger.warning(
                f"[LLM] rate limit | waiting {wait:.1f}s..."
            )

            time.sleep(wait)

        except APITimeoutError as e:
            last_error = e

            logger.warning(
                f"[LLM] timeout | attempt={attempt}"
            )

        except Exception as e:
            last_error = e

            logger.error(
                f"[LLM] unexpected error | "
                f"attempt={attempt} | {e}"
            )
            break

    raise RuntimeError(
        f"[LLM] all {MAX_RETRIES} attempts failed "
        f"for raw call | last_error={last_error}"
    )




def build_example_instance(schema: Type[T]) -> dict:
    """
    Build a sample JSON object from the Pydantic schema.
    Helps the model understand the expected output format.
    """
    example = {}

    for field_name, field_info in schema.model_fields.items():
        annotation = field_info.annotation

        if annotation is bool:
            example[field_name] = True
        elif annotation is str:
            example[field_name] = "example value"
        elif annotation is int:
            example[field_name] = 1
        elif annotation is float:
            example[field_name] = 1.0
        else:
            example[field_name] = "..."

    return example


def call_llm_structured(
    prompt: str,
    schema: Type[T],
    system: str = DEFAULT_SYSTEM,
    temperature: float = TEMPERATURE,
) -> T:
    """
    Call Groq LLM and return a validated Pydantic object.

    Groq JSON output is requested manually and validated
    using Pydantic.
    """
    client = get_client()
    last_error: Exception | None = None

    system_with_schema = (
        f"{system}\n\n"
        f"You must respond with ONLY valid JSON "
        f"that matches this schema:\n"
        f"{json.dumps(schema.model_json_schema(), indent=2)}\n\n"
        f"Example of a valid response:\n"
        f"{json.dumps(build_example_instance(schema), indent=2)}\n\n"
        f"Return actual values for this specific input. "
        f"Do NOT return the schema definition itself. "
        f"No explanation. No markdown. Just the JSON object."
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            limiter.acquire()

            logger.debug(
                f"[LLM] groq/{MODEL} | "
                f"schema={schema.__name__} | attempt={attempt}"
            )

            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": system_with_schema,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=temperature,
                max_tokens=MAX_TOKENS,
                reasoning_effort=REASONING_EFFORT,
            )

            raw = response.choices[0].message.content

            if not raw:
                raise ValueError("[LLM] Empty response received")

            raw = raw.strip()

            # Remove markdown JSON fences if present
            if raw.startswith("```"):
                raw = raw.split("```", 2)[1].strip()

                if raw.startswith("json"):
                    raw = raw[4:].strip()

            parsed = schema.model_validate_json(raw)

            logger.debug(
                f"[LLM] structured success | "
                f"schema={schema.__name__}"
            )

            return parsed

        except RateLimitError as e:
            last_error = e
            wait = backoff_wait(attempt, e)

            logger.warning(
                f"[LLM] rate limit | waiting {wait:.1f}s..."
            )

            time.sleep(wait)

        except APITimeoutError as e:
            last_error = e

            logger.warning(
                f"[LLM] timeout | attempt={attempt}"
            )

        except (json.JSONDecodeError, ValidationError) as e:
            last_error = e

            logger.warning(
                f"[LLM] JSON/schema validation failed | "
                f"attempt={attempt} | {e}"
            )

        except Exception as e:
            last_error = e

            logger.error(
                f"[LLM] unexpected error | "
                f"attempt={attempt} | {e}"
            )
            break

    raise RuntimeError(
        f"[LLM] all {MAX_RETRIES} attempts failed "
        f"for schema={schema.__name__} | "
        f"last_error={last_error}"
    )


