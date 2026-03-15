"""LLM Client - Groq API integration with robust error handling."""
import json
import requests
import time
from typing import Optional

from config import GROQ_API_KEY, GROQ_MODEL, GROQ_URL


class GroqClient:
    """Groq LLM client — free tier, fast inference."""

    def __init__(self):
        self.api_key = GROQ_API_KEY
        self.model   = GROQ_MODEL
        self.url     = GROQ_URL

        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Add it to your .env file."
            )

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def answer_with_context(
        self,
        query: str,
        docs: list,
        graph_context: Optional[dict] = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        retries: int = 3,
    ) -> dict:
        if not docs:
            return self._error_response("No context documents provided.")

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._build_system_prompt()},
                {"role": "user",   "content": self._build_user_prompt(query, docs, graph_context)},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }

        last_error = None
        for attempt in range(1, retries + 1):
            try:
                response = requests.post(self.url, headers=self.headers, json=payload, timeout=60)

                if response.status_code == 401:
                    return self._error_response("Invalid GROQ_API_KEY. Check your .env file.")
                if response.status_code == 429:
                    time.sleep(2 ** attempt)
                    last_error = "Rate limited by Groq API."
                    continue
                if response.status_code == 503:
                    time.sleep(2)
                    last_error = "Groq API temporarily unavailable."
                    continue
                if not response.ok:
                    return self._error_response(
                        f"Groq API error {response.status_code}: {response.text[:300]}"
                    )

                data    = response.json()
                choices = data.get("choices")
                if not choices or not isinstance(choices, list):
                    return self._error_response(f"Unexpected API response: {str(data)[:300]}")

                content = choices[0].get("message", {}).get("content", "").strip()
                if not content:
                    return self._error_response("Model returned an empty response.")

                return {
                    "answer": content,
                    "model":  data.get("model", self.model),
                    "usage":  data.get("usage", {}),
                    "error":  None,
                }

            except requests.exceptions.Timeout:
                last_error = f"Request timed out (attempt {attempt}/{retries})."
                time.sleep(1)
            except requests.exceptions.ConnectionError as exc:
                last_error = f"Connection error: {exc}"
                time.sleep(1)
            except (json.JSONDecodeError, KeyError, IndexError) as exc:
                return self._error_response(f"Failed to parse API response: {exc}")
            except Exception as exc:
                return self._error_response(f"Unexpected error: {exc}")

        return self._error_response(last_error or "All retry attempts failed.")

    @staticmethod
    def _build_system_prompt() -> str:
        return (
            "You are a precise, analytical assistant that answers questions "
            "strictly based on the provided context documents. "
            "If the answer is not contained in the context, say so clearly. "
            "Be concise, factual, and cite relevant details from the context."
        )

    @staticmethod
    def _build_user_prompt(query: str, docs: list, graph_context: Optional[dict]) -> str:
        context_parts = []
        for i, doc in enumerate(docs[:5], 1):
            text   = doc.get("text", "").strip()
            source = doc.get("source", "unknown")
            page   = doc.get("page", "?")
            if text:
                context_parts.append(f"[Document {i}] (source: {source}, page: {page})\n{text}")

        context_block = "\n\n".join(context_parts) if context_parts else "No context."

        graph_block = ""
        if graph_context:
            neighbours = graph_context.get("neighbors", [])
            if neighbours:
                graph_block = (
                    f"\n\nRelated entities from knowledge graph: "
                    f"{', '.join(str(n) for n in neighbours[:10])}"
                )

        return (
            f"Context:\n{context_block}"
            f"{graph_block}"
            f"\n\nQuestion: {query}"
            f"\n\nAnswer based only on the context above:"
        )

    @staticmethod
    def _error_response(message: str) -> dict:
        return {
            "answer": f"⚠️ Unable to generate an answer: {message}",
            "model":  "",
            "usage":  {},
            "error":  message,
        }