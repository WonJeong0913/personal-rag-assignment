"""배포 secrets 또는 로컬 입력 파일을 세션 메모리에서만 검색하는 Streamlit 앱."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import streamlit as st

from rag_core import (
    GEMINI_MODEL_DEFAULT,
    RagSession,
    build_embeddings,
    create_rag_session,
    dispose_rag_session,
    generate_with_gemini,
    generate_with_ollama,
    load_documents_from_path,
    load_documents_from_text,
    ollama_model_available,
    retrieve_documents,
    source_label,
)

LOGGER = logging.getLogger(__name__)

SESSION_KEY = "personal_rag_session"
HISTORY_KEY = "personal_rag_history"


def _secret_text(name: str) -> str:
    """값을 화면이나 로그에 표시하지 않고 Streamlit secrets의 문자열만 읽는다."""

    try:
        value = st.secrets.get(name, "")
    except (FileNotFoundError, KeyError):
        return ""
    return value.strip() if isinstance(value, str) else ""


def _has_profile_secret() -> bool:
    """PROFILE_DATA의 존재만 확인한다. 실제 원문은 인덱싱할 때 읽는다."""

    try:
        return "PROFILE_DATA" in st.secrets
    except FileNotFoundError:
        return False


def _current_session() -> RagSession | None:
    candidate = st.session_state.get(SESSION_KEY)
    return candidate if isinstance(candidate, RagSession) else None


def _replace_session(session: RagSession) -> None:
    dispose_rag_session(_current_session())
    st.session_state[SESSION_KEY] = session


def _clear_session() -> None:
    dispose_rag_session(_current_session())
    st.session_state.pop(SESSION_KEY, None)
    st.session_state.pop(HISTORY_KEY, None)


@st.cache_resource(show_spinner=False)
def _shared_embeddings() -> Any:
    """공개 모델 가중치만 worker 안에서 재사용한다. 개인 index는 여기에 저장하지 않는다."""

    return build_embeddings()


def _load_profile_documents() -> list:
    """Cloud의 PROFILE_DATA 또는 로컬 단일 입력 파일만 허용한다."""

    secret_profile = _secret_text("PROFILE_DATA")
    if secret_profile:
        return load_documents_from_text(secret_profile)
    local_profile = Path(__file__).parent / "data" / "private_profile.txt"
    if not local_profile.is_file():
        raise FileNotFoundError("입력 데이터가 설정되지 않았습니다.")
    return load_documents_from_path(local_profile)


def _ensure_rag_session() -> RagSession:
    session = _current_session()
    if session is not None:
        return session
    with st.spinner("자료를 준비하는 중입니다..."):
        session = create_rag_session(_load_profile_documents(), embedding_function=_shared_embeddings())
    _replace_session(session)
    return session


def _show_sources(documents: list) -> None:
    for number, document in enumerate(documents, 1):
        with st.expander(f"출처 {number}: {source_label(document)}"):
            st.text(document.page_content)


def _history() -> list[dict[str, Any]]:
    history = st.session_state.get(HISTORY_KEY)
    if not isinstance(history, list):
        history = []
        st.session_state[HISTORY_KEY] = history
    return history


def _render_history() -> None:
    for turn in _history():
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            _show_sources(turn["sources"])


def _generate_answer(backend: str, question: str, sources: list, api_key: str) -> str:
    if backend == "Ollama (로컬)":
        if not ollama_model_available():
            raise RuntimeError("로컬 Ollama와 필요한 모델을 찾을 수 없습니다.")
        return generate_with_ollama(question, sources)
    if not api_key:
        raise RuntimeError("배포용 Gemini API 키가 설정되지 않았습니다.")
    return generate_with_gemini(question, sources, api_key=api_key, model=GEMINI_MODEL_DEFAULT)


def main() -> None:
    st.set_page_config(page_title="개인 프로필 RAG", page_icon="👤")
    deployment_mode = _has_profile_secret()
    api_key = _secret_text("GOOGLE_API_KEY")
    if deployment_mode and not api_key:
        st.title("개인 프로필 RAG")
        st.error("배포 답변 설정을 확인해 주세요.")
        return
    st.title("개인 프로필 RAG")
    st.caption("질문에 답할 때 제공된 자료와 출처만 사용합니다.")

    with st.sidebar:
        backend = "Gemini" if api_key else "Ollama (로컬)"
        if st.button("대화와 현재 인덱스 지우기", use_container_width=True):
            _clear_session()
            st.rerun()

    _render_history()
    question = st.chat_input("제공된 자료에 관해 질문하세요")
    if not question:
        return

    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        try:
            session = _ensure_rag_session()
            sources = retrieve_documents(session, question)
            with st.spinner("답변을 준비하는 중입니다..."):
                answer = _generate_answer(backend, question, sources, api_key)
            st.write(answer)
            _show_sources(sources)
            _history().append({"question": question, "answer": answer, "sources": sources})
        except Exception as exc:
            # 예외 상세에는 원문 또는 provider 정보가 포함될 수 있으므로 표시하지 않는다.
            LOGGER.warning("personal_rag_request_failed type=%s", type(exc).__name__)
            st.error("답변을 준비하지 못했습니다. 설정을 확인해 주세요.")


if __name__ == "__main__":
    main()
