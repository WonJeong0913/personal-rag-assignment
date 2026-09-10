# 개인 프로필 RAG

제공된 개인 프로필의 13개 기준 사실과 100개 문답을 근거로 답하는 Streamlit 앱입니다. 검색 인덱스와 대화는 브라우저 세션의 메모리에만 존재하며, 디스크에 Chroma 데이터를 저장하지 않습니다.

## 로컬 실행

이 폴더에서 가상환경과 의존성을 준비합니다.

```bash
cd outputs/personal_rag
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

로컬 Ollama를 loopback 주소에서 시작합니다.

```bash
OLLAMA_HOST=127.0.0.1:11434 ollama serve
```

다른 터미널에서 앱을 로컬 주소로 실행합니다.

```bash
streamlit run app.py --server.address 127.0.0.1
```

기본 답변 엔진은 이 컴퓨터의 `qwen2.5:3b` Ollama 모델입니다. Ollama 요청은 `127.0.0.1:11434/api/chat`으로만 전송하며, 모델 응답 실패 시 답을 대신 만들어 내지 않습니다.

## 데이터와 검색

- 실제 입력 원문은 `data/private_profile.txt` 한 파일 또는 배포 secrets의 `PROFILE_DATA` 문자열에서만 읽습니다. `data/`와 `secrets.toml`은 Git에서 제외됩니다.
- 입력 파일의 `[기준 사실]`은 이름, 전화번호, 주소, 가족, 학력, 현재 재학 정보를 13개 canonical label로 둡니다. 이어지는 `Q1.`/`A1.`부터 `Q100.`/`A100.`은 모두 하나의 Q/A 문서로 보존됩니다.
- canonical 사실은 분야별 Document로, Q/A는 질문과 답을 합친 Document로 만듭니다. 일반 TXT/PDF는 `chunk_size=500`, `chunk_overlap=50`으로 분할합니다.
- CPU `jhgan/ko-sbert-nli` 임베딩과 Chroma `EphemeralClient`를 사용합니다. MMR 후보와 어휘 후보를 함께 보고, 단일 질문은 해당 Q/A를 우선하며 복합 질문은 요청된 각 분야의 canonical 근거를 우선합니다.
- Streamlit worker는 공개 임베딩 모델 가중치만 재사용합니다. private Chroma collection은 브라우저 세션마다 새로 만들고, 대화 삭제 시 해당 collection만 제거합니다.

`rag_core.parse_document_bytes()`는 UTF-8 TXT와 텍스트 추출 가능한 PDF를 메모리에서 읽을 수 있습니다. 앱 화면은 배포된 입력 자료 하나만 사용합니다.

## Streamlit Community Cloud 설정

배포 환경의 `.streamlit/secrets.toml`에 다음 키를 설정합니다. 저장소에는 [secrets.example.toml](.streamlit/secrets.example.toml)만 두고 실제 secrets 파일은 올리지 않습니다.

```toml
PROFILE_DATA = "배포할 입력 데이터 전체"
GOOGLE_API_KEY = "Gemini API key"
SITE_PASSWORD = "배포 접근 비밀번호"
```

- `PROFILE_DATA`를 설정하면 Cloud는 로컬 `data/` 파일 없이 같은 입력 형식을 사용합니다.
- `GOOGLE_API_KEY`가 있으면 앱은 Gemini를 자동으로 사용하며, 사용자는 키나 엔진을 고르지 않고 질문만 합니다. 이 경우 질문과 검색된 문맥이 Google API에 전송됩니다.
- `PROFILE_DATA`를 배포에 쓸 때 `SITE_PASSWORD`는 필수입니다. 비밀번호를 통과하기 전에는 입력 자료를 읽거나 인덱스를 만들고, 질문과 출처를 표시할 수 없습니다. 인증 상태는 현재 브라우저 세션 메모리에만 둡니다.

Cloud 플랫폼용 `.streamlit/config.toml`에는 bind address를 고정하지 않았습니다. 로컬 주소 고정은 위의 실행 명령에서만 합니다.

## 검증 명령

실제 개인 원문을 출력하지 않는 합성 테스트입니다.

```bash
cd outputs/personal_rag
pytest -q
```

검색 전수와 Ollama 실제 생성, Streamlit smoke는 별도 실행 환경에서 확인합니다.
