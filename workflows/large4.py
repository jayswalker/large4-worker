"""A passthrough chat workflow for Mistral Large 4 (Le Chonk)."""
from typing import Optional

from pydantic import BaseModel, Field

import mistralai.workflows as workflows
from mistralai.workflows import Depends
from mistralai.workflows.client import get_mistral_client

MODEL = "mistral-large-4"


class Large4Input(BaseModel):
    prompt: str = Field(..., description="The prompt to send to Mistral Large 4.")
    system_prompt: Optional[str] = Field(
        default="", description="Optional system prompt to steer the model."
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, gt=0)


class Large4Output(BaseModel):
    reply: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


@workflows.activity()
async def call_large4(
    prompt: str,
    system_prompt: Optional[str],
    temperature: float,
    max_tokens: Optional[int],
    client=Depends(get_mistral_client),
) -> Large4Output:
    """Send one prompt to Mistral Large 4 and return the reply."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    kwargs = {}
    if max_tokens:
        kwargs["max_tokens"] = max_tokens

    response = await client.chat.complete_async(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        **kwargs,
    )

    choice = response.choices[0]
    usage = response.usage
    return Large4Output(
        reply=choice.message.content or "",
        model=response.model or MODEL,
        input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(usage, "completion_tokens", 0) or 0,
    )


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
