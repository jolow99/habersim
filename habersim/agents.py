"""
LLM-backed agent implementations.

Includes a generic LLMAgent that works with any architecture,
and Anthropic/OpenAI client implementations.
"""

from __future__ import annotations
import json

from habersim.core import Action, LLMAgent, LLMClient, Perception


# ---------------------------------------------------------------------------
# Generic LLM Agent
# ---------------------------------------------------------------------------

class GenericLLMAgent(LLMAgent):
    """
    An agent that uses an LLM to decide actions given a perception.
    Works with any architecture — the architecture's ActionSpec list
    is serialised into the prompt as a tool-use schema.
    """

    def act(self, perception: Perception) -> Action:
        """Choose an action by prompting the LLM with the current perception."""
        prompt = self._build_prompt(perception)
        schema = self._build_schema(perception)

        try:
            result = self.llm.complete_json(
                prompt=prompt,
                schema=schema,
                system=self.system_prompt(),
            )
            return Action(
                agent_id=self.agent_id,
                action_type=result["action_type"],
                payload=result.get("payload", {}),
                reasoning=result.get("reasoning", ""),
            )
        except Exception as e:
            raise RuntimeError(f"LLM agent {self.agent_id} failed to produce valid action: {e}")

    def _build_prompt(self, perception: Perception) -> str:
        lines = [
            f"## Deliberation Topic\n{perception.topic}",
            "",
            "## Current State",
            json.dumps(perception.context, indent=2, default=str),
            "",
            "## Available Actions",
        ]
        for spec in perception.available_actions:
            lines.append(f"- **{spec.name}**: {spec.description}")
            if spec.parameters:
                lines.append(f"  Parameters: {json.dumps(spec.parameters)}")
        lines += [
            "",
            "Choose one action and respond in the JSON format specified.",
            "Include a 'reasoning' field explaining your thinking.",
        ]
        return "\n".join(lines)

    def _build_schema(self, perception: Perception) -> dict:
        action_names = [s.name for s in perception.available_actions]
        return {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Your step-by-step reasoning before choosing an action.",
                },
                "action_type": {
                    "type": "string",
                    "enum": action_names,
                    "description": "The action you choose to take.",
                },
                "payload": {
                    "type": "object",
                    "description": "Parameters for the chosen action.",
                },
            },
            "required": ["reasoning", "action_type", "payload"],
        }


# ---------------------------------------------------------------------------
# Anthropic LLM client
# ---------------------------------------------------------------------------

class AnthropicClient(LLMClient):
    """
    LLMClient backed by the Anthropic API (claude-* models).
    Requires: pip install anthropic
    """

    def __init__(self, model: str = "claude-opus-4-6", api_key: str | None = None):
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic")

        self.model = model
        self._client = anthropic.Anthropic(api_key=api_key)

    def complete(self, prompt: str, system: str = "", **kwargs) -> str:
        """Send a prompt to the Anthropic API and return the text response."""
        response = self._client.messages.create(
            model=self.model,
            max_tokens=kwargs.get("max_tokens", 1024),
            system=system or "You are a helpful assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def complete_json(self, prompt: str, schema: dict, system: str = "", **kwargs) -> dict:
        """Send a prompt requesting JSON output conforming to the given schema."""
        json_instruction = (
            f"\n\nRespond ONLY with valid JSON conforming to this schema:\n"
            f"{json.dumps(schema, indent=2)}\n"
            "Do not include any text outside the JSON object."
        )
        full_prompt = prompt + json_instruction
        raw = self.complete(full_prompt, system=system, **kwargs)

        # Strip markdown code fences if present
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
            raw = raw.rsplit("```", 1)[0]

        return json.loads(raw)


# ---------------------------------------------------------------------------
# OpenAI LLM client
# ---------------------------------------------------------------------------

class OpenAIClient(LLMClient):
    """
    LLMClient backed by the OpenAI API.
    Requires: pip install openai
    """

    def __init__(self, model: str = "gpt-4o", api_key: str | None = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")

        self.model = model
        self._client = OpenAI(api_key=api_key)

    def complete(self, prompt: str, system: str = "", **kwargs) -> str:
        """Send a prompt to the OpenAI API and return the text response."""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system or "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=kwargs.get("max_tokens", 1024),
        )
        return response.choices[0].message.content

    def complete_json(self, prompt: str, schema: dict, system: str = "", **kwargs) -> dict:
        """Send a prompt requesting JSON output conforming to the given schema."""
        json_instruction = (
            f"\n\nRespond ONLY with valid JSON conforming to this schema:\n"
            f"{json.dumps(schema, indent=2)}\n"
            "Do not include any text outside the JSON object."
        )
        full_prompt = prompt + json_instruction
        raw = self.complete(full_prompt, system=system, **kwargs)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
            raw = raw.rsplit("```", 1)[0]
        return json.loads(raw)
