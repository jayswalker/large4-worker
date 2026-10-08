"""Large 4 chat workflow — diagnostic build (temporary).

Probes the SA-token client and two models under hard timeouts, and returns
either Le Chonk's reply or a full text report of what failed.
"""
import asyncio
import traceback
from datetime import timedelta
from typing import Optional

from pydantic import BaseModel, Field

import mistralai.workflows as workflows
from mistralai.workflows.client import get_mistral_client

MODEL = "mistral-large-4"
CHEAP_MODEL = "mistral-small-latest"


class Large4Input(BaseModel):
    prompt: str = Field(..., description="The prompt to send to Mistral Large 4.")
    system_prompt: Optional[str] = Field(default="", description="Optional system prompt to steer the model.")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, gt=0)


class Large4Output(BaseModel):
    reply: str = ""
    model: str = MODEL
    input_tokens: int = 0
    output_tokens: int = 0
    error: Optional[str] = None


async def _probe(label: str, factory, timeout_s: float) -> str:
    """Run one awaitable under a hard timeout and describe the outcome."""
    try:
        await asyncio.wait_for(factory(), timeout=timeout_s)
        return f"{label}: OK"
    except asyncio.TimeoutError:
        return f"{label}: TIMED OUT after {timeout_s}s (no response, no error)"
    except Exception as exc:
        detail = str(exc).replace("\n", " ")[:600]
        return f"{label}: {type(exc).__name__}: {detail}"


@workflows.activity(start_to_close_timeout=timedelta(seconds=110))
async def call_large4(
    prompt: str,
    system_prompt: Optional[str],
    temperature: float,
    max_tokens: Optional[int],
) -> Large4Output:
    """Attempt the Large 4 call; on failure, return a diagnostic report."""
    report: list = []

    try:
        client = get_mistral_client()
        report.append("client: constructed OK")
    except Exception:
        return Large4Output(error="get_mistral_client() failed:\n" + traceback.format_exc()[-1500:])

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    kwargs = {}
    if max_tokens:
        kwargs["max_tokens"] = max_tokens

    # Probe 0: can the SA token list models at all?
    if hasattr(client.models, "list_async"):
        report.append(await _probe("models.list", client.models.list_async, 20))
    else:
        report.append("models.list_async: not available in SDK")

    # Probe 1: the real call to Large 4.
    async def call_large4_model():
        return await client.chat.complete_async(
            model=MODEL, messages=messages, temperature=temperature, **kwargs
        )

    try:
        response = await asyncio.wait_for(call_large4_model(), timeout=45)
    except asyncio.TimeoutError:
        report.append("chat mistral-large-4: TIMED OUT after 45s")
        response = None
    except Exception as exc:
        report.append(f"chat mistral-large-4: {type(exc).__name__}: {str(exc)[:600]}")
        response = None

    if response is not None:
        choice = response.choices[0]
        usage = response.usage
        return Large4Output(
            reply=choice.message.content or "",
            model=response.model or MODEL,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )

    # Probe 2: does ANY model answer with this client?
    async def call_cheap_model():
        return await client.chat.complete_async(
            model=CHEAP_MODEL,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=20,
        )

    report.append(await _probe(f"chat {CHEAP_MODEL}", call_cheap_model, 25))

    text = "\n".join(report)
    print("LARGE4 DEBUG REPORT:\n" + text)
    return Large4Output(error=text)


@workflows.workflow.define(
    name="large4-chat",
    workflow_display_name="Large 4 Chat",
    workflow_description="Send a prompt to Mistral Large 4 and return its reply.",
)
class Large4Workflow:
    @workflows.workflow.entrypoint
    async def run(self, input: Large4Input) -> Large4Output:
        return await call_large4(
            prompt=input.prompt,
            system_prompt=input.system_prompt,
            temperature=input.temperature,
            max_tokens=input.max_tokens,
        )
