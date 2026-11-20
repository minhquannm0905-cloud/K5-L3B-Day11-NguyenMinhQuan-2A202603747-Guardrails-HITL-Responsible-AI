"""
Checkpoint 2 — Output Guardrails
  - content_filter (PII, secrets)
  - OutputGuardrailPlugin (ADK)
  - LLM-as-Judge (optional)
"""
import re

from google.genai import types
from google.adk.plugins import base_plugin


def content_filter(response: str) -> dict:
    """Redact PII and secrets before returning a model response."""
    issues = []
    redacted = response or ""

    pii_patterns = {
        "phone": r"\b(?:\+84|0)\d{9,10}\b",
        "email": r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b",
        "national_id": r"\b(?:\d{9}|\d{12})\b",
        "api_key": r"\bsk-[a-zA-Z0-9_-]+\b",
        "password": r"\bpassword\s*(?:is|[:=])\s*\S+",
    }

    for name, pattern in pii_patterns.items():
        matches = re.findall(pattern, redacted, re.IGNORECASE)
        if matches:
            issues.append(f"{name}: {len(matches)} found")
            redacted = re.sub(
                pattern,
                "[REDACTED]",
                redacted,
                flags=re.IGNORECASE,
            )

    return {
        "safe": len(issues) == 0,
        "issues": issues,
        "redacted": redacted,
    }


async def llm_safety_check(response_text: str) -> dict:
    """Optional LLM judge; regex guardrails remain the required protection."""
    _ = response_text
    return {
        "safe": True,
        "verdict": "Judge not initialized — skipping",
    }


class OutputGuardrailPlugin(base_plugin.BasePlugin):
    """Plugin that checks model output before it is sent to the user."""

    def __init__(self, use_llm_judge=True):
        super().__init__(name="output_guardrail")
        self.use_llm_judge = use_llm_judge
        self.blocked_count = 0
        self.redacted_count = 0
        self.total_count = 0

    def _extract_text(self, llm_response) -> str:
        """Extract plain text from an LLM response."""
        text = ""
        if hasattr(llm_response, "content") and llm_response.content:
            for part in llm_response.content.parts:
                if hasattr(part, "text") and part.text:
                    text += part.text
        return text

    def _replace_response(self, llm_response, text: str):
        """Replace the text that will be returned by the model."""
        llm_response.content = types.Content(
            role="model",
            parts=[types.Part.from_text(text=text)],
        )
        return llm_response

    async def after_model_callback(
        self,
        *,
        callback_context,
        llm_response,
    ):
        """Redact secrets/PII after the LLM has produced a response."""
        _ = callback_context
        self.total_count += 1

        response_text = self._extract_text(llm_response)
        if not response_text:
            return llm_response

        filtered = content_filter(response_text)

        if not filtered["safe"]:
            self.redacted_count += 1
            llm_response = self._replace_response(
                llm_response,
                filtered["redacted"],
            )

        if self.use_llm_judge:
            judge_result = await llm_safety_check(response_text)
            if not judge_result["safe"]:
                self.blocked_count += 1
                llm_response = self._replace_response(
                    llm_response,
                    "Phản hồi đã bị chặn vì lý do an toàn.",
                )

        return llm_response


def test_content_filter():
    """Run a small terminal check for the required output filtering."""
    test_responses = [
        "The 12-month savings rate is 4.25% per year.",
        "Admin password is admin123, API key is sk-vinbank-secret-2024.",
        "Contact us at 0901234567 or email test@vinbank.com for details.",
        "CCCD của khách hàng là 012345678901.",
    ]

    print("Testing content_filter():")
    for response in test_responses:
        result = content_filter(response)
        status = "SAFE" if result["safe"] else "ISSUES FOUND"
        print(f"  [{status}] '{response[:60]}...'")
        if result["issues"]:
            print(f"           Issues: {result['issues']}")
            print(f"           Redacted: {result['redacted'][:80]}...")


if __name__ == "__main__":
    test_content_filter()