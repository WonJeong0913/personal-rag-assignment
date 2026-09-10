from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rag_core


class TinyEmbeddings:
    """네트워크 모델을 쓰지 않고 실제 Ephemeral Chroma 테스트를 위한 합성 임베딩."""

    @staticmethod
    def _vector(text: str) -> list[float]:
        buckets = [0.0] * 8
        for character in text:
            buckets[ord(character) % len(buckets)] += 1.0
        magnitude = sum(value * value for value in buckets) ** 0.5 or 1.0
        return [value / magnitude for value in buckets]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def synthetic_profile() -> str:
    values = {field: f"synthetic-{number}" for number, field in enumerate(rag_core.CANONICAL_FIELDS, 1)}
    lines = ["[기준 사실]", *(f"{field}: {values[field]}" for field in rag_core.CANONICAL_FIELDS), "", "[100개 문답]"]
    for number in range(1, 101):
        field = rag_core.CANONICAL_FIELDS[(number - 1) % len(rag_core.CANONICAL_FIELDS)]
        label = "현재 재학 학교" if field == "현재학교" else field
        lines.extend((
            f"Q{number}. 대상자의 {label} 정보를 알려주세요 {number}.",
            f"A{number}. {field}: {values[field]}.",
            "",
        ))
    return "\n".join(lines)


@pytest.fixture
def parsed_documents() -> list:
    return rag_core.parse_document_bytes("synthetic.txt", synthetic_profile().encode("utf-8"))


def test_100_unique_qa_and_canonical_facts_stay_in_one_document(parsed_documents: list) -> None:
    canonical = [document for document in parsed_documents if document.metadata["record_type"] == "canonical"]
    qa_documents = [document for document in parsed_documents if document.metadata["record_type"] == "qa"]

    assert len(canonical) == len(rag_core.CANONICAL_FIELDS)
    assert {document.metadata["fields"] for document in canonical} == set(rag_core.CANONICAL_FIELDS)
    assert len(qa_documents) == 100
    assert len({document.page_content.splitlines()[0] for document in qa_documents}) == 100
    assert all(document.page_content.startswith("Q") and "\nA" in document.page_content for document in qa_documents)
    assert rag_core.split_profile_documents(parsed_documents) == parsed_documents


def test_ephemeral_indexes_are_separate_and_disposal_is_scoped(parsed_documents: list) -> None:
    first = rag_core.create_rag_session(parsed_documents, embedding_function=TinyEmbeddings())
    second = rag_core.create_rag_session(parsed_documents, embedding_function=TinyEmbeddings())
    try:
        assert first.collection_name != second.collection_name
        assert first.vector_store._collection.count() == len(parsed_documents)
        assert second.vector_store._collection.count() == len(parsed_documents)
        rag_core.dispose_rag_session(first)
        assert second.vector_store._collection.count() == len(parsed_documents)
    finally:
        rag_core.dispose_rag_session(first)
        rag_core.dispose_rag_session(second)


def test_simple_and_compound_retrieval_keep_field_evidence_diverse(parsed_documents: list) -> None:
    session = rag_core.create_rag_session(parsed_documents, embedding_function=TinyEmbeddings())
    try:
        simple = rag_core.retrieve_documents(session, "대상자의 전화번호 정보를 알려주세요.")
        assert simple[0].metadata["record_type"] == "qa"
        assert "전화번호" in simple[0].metadata["fields"]
        signatures = [document.metadata.get("answer_signature") for document in simple if document.metadata.get("answer_signature")]
        assert len(signatures) == len(set(signatures))

        compound = rag_core.retrieve_documents(session, "대상자의 이름, 전화번호, 주소를 함께 알려주세요.", k=3)
        assert {"이름", "전화번호", "주소"} <= {field for document in compound for field in document.metadata["fields"].split("|")}
        assert all(document.metadata["record_type"] == "canonical" for document in compound)
    finally:
        rag_core.dispose_rag_session(session)


def test_profile_and_school_abbreviation_intents_select_the_right_fields() -> None:
    all_fields = tuple(rag_core.CANONICAL_FIELDS)
    for question in (
        "대상자에 대해 알려주세요.",
        "대상자 프로필을 알려주세요.",
        "대상자를 자기소개해 주세요.",
        "대상자 소개해 주세요.",
    ):
        assert rag_core.detect_question_fields(question) == all_fields

    school_fields = ("졸업초등학교", "졸업중학교", "졸업고등학교")
    for question in (
        "대상자가 졸업한 초중고를 알려주세요.",
        "초·중·고 학력을 알려주세요.",
        "초 중 고를 확인해 주세요.",
        "대상자가 졸업한 학교가 어디야?",
        "대상자가 졸업했던 학교를 알려주세요.",
        "대상자는 어느 학교를 졸업했나요?",
    ):
        assert rag_core.detect_question_fields(question) == school_fields

    assert rag_core.detect_question_fields("대상자의 가족을 소개해 주세요.") == ("어머니", "아버지", "동생")
    assert rag_core.detect_question_fields("대상자의 학과에 대해 알려주세요.") == ("학과",)


def test_full_profile_request_preserves_all_13_canonical_evidence(parsed_documents: list) -> None:
    session = rag_core.create_rag_session(parsed_documents, embedding_function=TinyEmbeddings())
    try:
        retrieved = rag_core.retrieve_documents(session, "대상자에 대해 알려주세요.")
        assert len(retrieved) == len(rag_core.CANONICAL_FIELDS)
        assert all(document.metadata["record_type"] == "canonical" for document in retrieved)
        assert {document.metadata["fields"] for document in retrieved} == set(rag_core.CANONICAL_FIELDS)
    finally:
        rag_core.dispose_rag_session(session)


def test_unprovided_information_is_explicit_in_generation_prompt() -> None:
    document = rag_core.parse_document_bytes("synthetic.txt", synthetic_profile().encode("utf-8"))[0]
    messages = rag_core.build_generation_messages("대상자의 생일은 언제인가요?", [document])

    assert rag_core.explicitly_unprovided_question("대상자의 생년월일은 언제인가요?")
    assert rag_core.ABSTENTION_TEXT in messages[0].content
    assert "성별" in messages[0].content
    assert json.loads(messages[1].content)["question"] == "대상자의 생일은 언제인가요?"


def test_ollama_generation_uses_fixed_loopback_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class Response:
        def read(self) -> bytes:
            return b'{"message":{"content":"synthetic answer"}}'

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

    def fake_urlopen(request: Any, timeout: int) -> Response:
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return Response()

    document = rag_core.parse_document_bytes("synthetic.txt", synthetic_profile().encode("utf-8"))[0]
    monkeypatch.setattr(rag_core, "urlopen", fake_urlopen)

    assert rag_core.generate_with_ollama("합성 질문", [document]) == "synthetic answer"
    assert captured["url"] == rag_core.OLLAMA_URL
    assert captured["timeout"] == rag_core.OLLAMA_TIMEOUT_SECONDS
    assert captured["body"]["model"] == rag_core.OLLAMA_MODEL_DEFAULT
    assert captured["body"]["stream"] is False
    assert captured["body"]["options"] == {"temperature": 0, "num_predict": rag_core.OLLAMA_NUM_PREDICT}


def test_password_comparison_accepts_korean_text() -> None:
    assert rag_core.verify_site_password("합성-암호", "합성-암호")
    assert not rag_core.verify_site_password("합성-암호", "다른-암호")
