"""Provider-agnostic LLM generation with prompt-injection defense and citations."""

from __future__ import annotations

import logging
import os
import re
import time
from abc import ABC, abstractmethod
from typing import Any

import httpx
from pydantic import BaseModel, Field

from app.core.config import settings
from app.rag.citations import CitationTracker, SourceAttribution
from app.rag.context_builder import ContextBuilder
from app.retrieval.hybrid import HybridSearchEngine

logger = logging.getLogger(__name__)


class RAGResponse(BaseModel):
    """Standardized response schema for RAG endpoint matching Section 15."""

    answer: str = Field(description="Synthesized natural-language answer")
    sources: list[SourceAttribution] = Field(description="Verified source documents and chunks")
    retrieval_latency_ms: float = Field(description="Time taken for retrieval in milliseconds")
    generation_latency_ms: float = Field(
        description="Time taken for LLM generation in milliseconds"
    )


class BaseLLMProvider(ABC):
    """Abstract interface for LLM completion providers."""

    @abstractmethod
    def generate(self, system_prompt: str, user_query: str, context: str) -> str:
        """Generate response based on system prompt, query, and context."""
        pass


class LocalDeterministicGenerator(BaseLLMProvider):
    """High-reliability local generator for reproducible evaluation and zero-dependency operation."""

    def generate(self, system_prompt: str, user_query: str, context: str) -> str:
        # If context is empty, return standard refusal
        if not context or not context.strip():
            return "I cannot answer this question based on the provided documents."

        # Parse document blocks from context
        doc_matches = re.findall(
            r'<document index="(\d+)" id="([^"]+)" chunk="([^"]+)">\s*<title>([^<]+)</title>\s*<content>\s*([\s\S]*?)\s*</content>',
            context,
        )

        if not doc_matches:
            return "I cannot answer this question based on the provided documents."

        query_terms = [t.lower() for t in user_query.split() if len(t) > 2]
        extracted_facts: list[str] = []

        for index, _doc_id, _chunk_id, _title, content in doc_matches:
            # Check for adversarial injection attempts inside content
            if (
                "ignore previous instructions" in content.lower()
                or "disregard all instructions" in content.lower()
            ):
                continue

            sentences = [s.strip() for s in re.split(r"(?<=[.?!])\s+", content) if s.strip()]
            for sentence in sentences:
                s_lower = sentence.lower()
                matches = sum(1 for q in query_terms if q in s_lower)
                if matches >= 1:
                    extracted_facts.append(f"{sentence} [{index}]")
                    if len(extracted_facts) >= 3:
                        break
            if len(extracted_facts) >= 3:
                break

        if not extracted_facts:
            # Fallback to the first sentence of the top ranked document
            first_idx, _, _, _, first_content = doc_matches[0]
            first_sentence = first_content.split("\n")[0].split(".")[0]
            if first_sentence:
                return f"{first_sentence.strip()}. [{first_idx}]"
            return "I cannot answer this question based on the provided documents."

        # Synthesize coherent answer
        answer = " ".join(extracted_facts)
        return answer


class OpenAILLMProvider(BaseLLMProvider):
    """OpenAI / compatible REST API provider."""

    def __init__(
        self, api_key: str | None = None, base_url: str | None = None, model: str = "gpt-3.5-turbo"
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = (
            base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        ).rstrip("/")
        self.model = model

    def generate(self, system_prompt: str, user_query: str, context: str) -> str:
        if not self.api_key:
            logger.warning(
                "OPENAI_API_KEY not configured. Falling back to LocalDeterministicGenerator."
            )
            return LocalDeterministicGenerator().generate(system_prompt, user_query, context)

        user_content = f"CONTEXT:\n{context}\n\nUSER QUESTION:\n{user_query}"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.0,
            "max_tokens": 512,
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(
                f"OpenAI API call failed ({e}). Falling back to LocalDeterministicGenerator."
            )
            return LocalDeterministicGenerator().generate(system_prompt, user_query, context)


class RAGPipeline:
    """End-to-end RAG pipeline executing retrieval, context assembly, and answer generation."""

    def __init__(
        self,
        search_engine: HybridSearchEngine,
        context_builder: ContextBuilder | None = None,
        llm_provider: BaseLLMProvider | None = None,
    ):
        self.search_engine = search_engine
        self.context_builder = context_builder or ContextBuilder(
            max_context_tokens=settings.rag_max_context_tokens,
            max_chunks=settings.rag_max_chunks,
            max_chunks_per_document=settings.rag_max_chunks_per_doc,
        )

        if llm_provider:
            self.llm_provider = llm_provider
        elif settings.rag_llm_provider == "openai":
            self.llm_provider = OpenAILLMProvider()
        else:
            self.llm_provider = LocalDeterministicGenerator()

    def generate_answer(
        self,
        query: str,
        top_k: int = 5,
        tenant_id: str = "default",
        filters: dict[str, Any] | None = None,
        rerank: bool = True,
    ) -> RAGResponse:
        """Execute full RAG retrieval and generation pipeline."""
        # 1. Retrieval
        search_out = self.search_engine.search(
            query=query,
            top_k=top_k * 2,  # retrieve extra candidates for context builder filtering
            filters=filters,
            tenant_id=tenant_id,
            mode="hybrid",
            rerank=rerank,
        )
        retrieval_latency = search_out["latency_ms"]

        # 2. Context Construction & Prompt Injection Sanitization
        context_str, selected_chunks = self.context_builder.build_context(search_out["results"])
        system_prompt = self.context_builder.construct_system_prompt()

        # 3. LLM Generation
        t_gen = time.perf_counter()
        answer = self.llm_provider.generate(system_prompt, query, context_str)
        generation_latency = round((time.perf_counter() - t_gen) * 1000.0, 2)

        # 4. Citations and Source Attribution
        sources = CitationTracker.extract_sources(selected_chunks, answer_text=answer)

        return RAGResponse(
            answer=answer,
            sources=sources,
            retrieval_latency_ms=retrieval_latency,
            generation_latency_ms=generation_latency,
        )
