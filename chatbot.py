import streamlit as st
import os
from dotenv import load_dotenv
import pypdf

# 2026년형 완전 표준 코어 모듈 및 멀티 쿼리 검색기
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser, SimpleJsonOutputParser


# .env 파일 로드
load_dotenv()

st.set_page_config(page_title="하이브리드 문서 요정", page_icon="🧚", layout="wide")


# =========================================================================
# 🎨 보완된 스타일 커스텀: HTML 카드 제거 및 실제 업로더를 드롭존 디자인으로 개조
# =========================================================================

CSS_STYLE = r"""
<style>
    body, .stApp { background: #f4f7fb; }
    .block-container { padding-top: 3rem; padding-bottom: 2rem; }
    .page-card, .panel-card, .section-card { background: #ffffff; border-radius: 24px; padding: 24px; box-shadow: 0 24px 60px rgba(24, 39, 75, 0.08); border: 1px solid #e2e8f0; }
    .page-card { margin-bottom: 1.5rem; }
    .section-card { background: #f8fbff; border: 1px solid #dbeafe; padding: 18px; margin-top: 18px; }
    .panel-divider { height: 1px; background: #e2e8f0; margin: 24px 0; border: none; }
    .app-title { font-size: 2.6rem; font-weight: 700; margin: 0; }
    .app-subtitle { color: #525f7f; font-size: 1.1rem; margin-top: 0.35rem; }
    .section-title { font-size: 1.3rem; font-weight: 700; margin-bottom: 0.75rem; }
    .section-label { display: inline-flex; align-items: center; gap: 0.6rem; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.08em; color: #475569; margin-bottom: 0.85rem; }
    .section-subtitle { font-size: 1rem; font-weight: 600; color: #0f172a; margin-bottom: 0.5rem; }
    .supported-file { background: #eef4ff; color: #1f4ed8; padding: 12px 16px; border-radius: 14px; display: inline-block; margin-bottom: 8px; }
    .upload-hint { color: #64748b; font-size: 0.96rem; line-height: 1.6; }
    
    /* ⭐️ [핵심 보완] 순정 파일 업로더 영역을 커스텀 드롭존 디자인으로 변딩 */
    [data-testid="stFileUploader"] {
        padding: 0 !important;
        margin-top: 8px !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background: #ffffff !important;
        border: 2px dashed #93c5fd !important;
        border-radius: 22px !important;
        padding: 32px 16px !important;
        text-align: center !important;
    }
    /* 내부 업로드 버튼 스타일 고도화 */
    [data-testid="stFileUploaderDropzone"] button {
        background-color: #eef4ff !important;
        color: #1f4ed8 !important;
        border: 1px solid #bfdbfe !important;
        border-radius: 14px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 600 !important;
    }
    [data-testid="stFileUploaderDropzone"] button:hover {
        background-color: #dbeafe !important;
    }
    
    /* ⭐️ [추가] 사이드바 모드 선택을 토글 버튼 스타일로 변경 */
    div[role="radiogroup"] {
        display: flex;
        background-color: #e2e8f0;
        border-radius: 18px;
        padding: 4px;
        margin: 0.5rem 0 1rem 0;
    }
    div[role="radiogroup"] label { /* 각 라디오 옵션의 라벨 */
        flex: 1;
        text-align: center;
        padding: 8px 0;
        border-radius: 14px;
        font-weight: 600;
        font-size: 0.9rem;
        cursor: pointer;
        transition: all 0.2s ease-in-out;
        color: #475569;
    }
    /* Streamlit 라디오 버튼의 기본 구조 `label > input` 을 이용 */
    div[role="radiogroup"] label:has(input:checked) {
        background-color: #ffffff;
        color: #1f4ed8;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    /* 실제 라디오 input 요소는 숨김 */
    div[role="radiogroup"] input {
        display: none;
    }

    /* ⭐️ [추가] 예상 질문 카드 스타일 */
    .prediction-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 20px;
        text-align: center;
        cursor: pointer;
        transition: all 0.2s ease-in-out;
        height: 100%;
    }
    .prediction-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px rgba(31, 78, 216, 0.1);
        border-color: #bfdbfe;
    }
    .prediction-card-selected {
        background: #eef4ff;
        border-color: #1d4ed8;
        transform: translateY(-4px);
        box-shadow: 0 12px 24px rgba(31, 78, 216, 0.1);
    }
    .prediction-question {
        font-weight: 600;
        color: #1e293b;
        font-size: 1rem;
        margin: 0;
    }

    .stButton>button { border-radius: 999px; padding: 0.95rem 1.35rem; font-weight: 600; }
    .stTextInput>div>div>input { border-radius: 14px; }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { color: #102a43; }
</style>
"""

# 상단 헤더 영역 (Base64로 인코딩된 요정 이미지를 src에 주입하세요)
MAIN_HEADER_HTML = r"""
<div style="display:flex; align-items:center; gap: 1.5rem; margin-bottom: 1.5rem;">
  <img src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7" width="80">
  <div>
    <p class="app-title">문서 요정과 대화해보세요!</p>
    <p class="app-subtitle">필요에 따라 고정된 로컬 DB를 선택하거나 새로운 파일을 직접 업로드하여 대화할 수 있습니다.</p>
  </div>
</div>
"""

# 사이드바 모드 설정 헤더
SIDEBAR_MODE_HEADER_HTML = r"""
<div class="page-card" style="padding: 20px; margin-bottom: 15px;">
  <div class="section-label">MODE SELECT</div>
  <div class="section-title" style="font-size: 1.15rem; margin: 0;">🤖 챗봇 모드 설정</div>
</div>
"""

# [모드 A] 고정 문서 DB (Chroma) 카드 (잘려 보이던 가짜 업로드 디자인 제거)
SIDEBAR_CHROMA_HTML = r"""
<div class="page-card" style="border-radius: 24px; padding: 24px; box-shadow: 0 10px 40px rgba(24, 39, 75, 0.04); margin-bottom: 0;">
  <div class="section-label">STEP 1</div>
  <div class="section-title">💾 고정 로컬 DB 로드</div>
  <p class="upload-hint">ingest.py를 통해 로컬에 구축된 chroma_db 스토리지를 불러옵니다.</p>
  <div class="panel-divider"></div>
</div>
"""

# [모드 B] 실시간 파일 업로드 카드 (잘려 보이던 가짜 업로드 디자인 제거)
SIDEBAR_UPLOAD_HTML = r"""
<div class="page-card" style="border-bottom-left-radius: 0; border-bottom-right-radius: 0; box-shadow: 0 10px 40px rgba(24, 39, 75, 0.04); margin-bottom: 0; border-bottom: none;">
  <div class="section-label">STEP 1</div>
  <div class="section-title">📂 실시간 문서 업로드</div>
  <p class="upload-hint">새로운 PDF 또는 TXT 문서를 추가하여 즉석에서 분석합니다.</p>
  <div class="panel-divider"></div>
  <div class="section-card">
    <div class="section-subtitle">지원되는 파일 형식</div>
    <div class="supported-file">.pdf</div>
    <div class="supported-file">.txt</div>
  </div>
  <div style="margin-top: 20px; font-size: 0.95rem; font-weight: 600; color: #0f172a;">업로드</div>
</div>
"""

# 사이드바 하단 STEP 2 카드
SIDEBAR_STEP2_HTML = r"""
<div class="page-card" style="padding: 20px; margin-top: 15px;">
  <div class="section-label">STEP 2</div>
  <div class="section-title" style="font-size: 1.15rem; margin-bottom: 4px;">🔎 실시간 대화 가능</div>
  <p class="upload-hint">현재 선택된 모드의 데이터를 바탕으로 요정과 자유롭게 질의응답을 나눕니다.</p>
</div>
"""
st.markdown(CSS_STYLE, unsafe_allow_html=True)

# =========================================================================
# ⭐️ [핵심 추가] 예상 질문 클릭 처리 로직
# =========================================================================
# URL 쿼리 파라미터를 사용하여 클릭 상태를 관리하고, 클릭 시 채팅 메시지를 추가합니다.
params = st.query_params
if "select_prediction" in params:
    new_index_str = params.get("select_prediction")
    # 이전에 클릭한 질문과 다른 질문을 클릭했을 때만 채팅 기록에 추가
    if st.session_state.get("selected_prediction_index") != new_index_str:
        st.session_state.selected_prediction_index = new_index_str
        try:
            index = int(new_index_str)
            # 미리 생성해둔 Q&A 세트를 가져와 채팅 기록에 추가
            qa_pair = st.session_state.qa_predictions[index]
            st.session_state.messages.append({"role": "user", "content": qa_pair["question"]})
            st.session_state.messages.append({"role": "assistant", "content": qa_pair["answer"]})
            st.rerun() # 채팅 기록을 즉시 업데이트하기 위해 재실행
        except (ValueError, IndexError, TypeError):
            # 잘못된 인덱스나 데이터가 없는 경우 무시
            st.session_state.selected_prediction_index = None

# 화면 구성 시작
st.markdown(MAIN_HEADER_HTML, unsafe_allow_html=True)

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "local_vector_store" not in st.session_state:
    st.session_state.local_vector_store = None
if "previous_mode" not in st.session_state:
    st.session_state.previous_mode = "📌 고정 문서 DB (chroma_db)"
if "qa_predictions" not in st.session_state:
    st.session_state.qa_predictions = None
if "selected_prediction_index" not in st.session_state:
    st.session_state.selected_prediction_index = None
if "last_uploaded_name" not in st.session_state:
    st.session_state.last_uploaded_name = None
if "active_store" not in st.session_state:
    st.session_state.active_store = None

# 임베딩 모델 정의
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

with st.sidebar:
    st.markdown(SIDEBAR_MODE_HEADER_HTML, unsafe_allow_html=True)
    # ⭐️ [수정] Selectbox를 가로형 Radio 버튼으로 변경하여 토글처럼 보이게 함
    chat_mode = st.radio(
        "사용할 데이터 소스를 골라주세요.",
        ["📌 고정 문서 DB (chroma_db)", "📂 실시간 파일 업로드"],
        label_visibility="collapsed",
        horizontal=True,
    )
    
    if chat_mode != st.session_state.previous_mode:
        st.session_state.messages = []
        st.session_state.previous_mode = chat_mode
        # ⭐️ [수정] 모드 변경 시, 이전 모드의 예상 질문이 남아있지 않도록 초기화합니다.
        st.session_state.qa_predictions = None
        st.session_state.selected_prediction_index = None
        st.rerun()

    # 모드에 따른 활성 벡터 저장소(active_store) 결정
    active_store = None

    # [모드 A] 고정 문서 DB (Chroma)
    if chat_mode == "📌 고정 문서 DB (chroma_db)":
        st.markdown(SIDEBAR_CHROMA_HTML, unsafe_allow_html=True)
        if os.path.exists("./chroma_db"):
            if st.session_state.local_vector_store is None:
                st.session_state.local_vector_store = Chroma(
                    persist_directory="./chroma_db",
                    embedding_function=embeddings_model
                )
            st.success("✅ 로컬 chroma_db 폴더 연결 완료!")
        else:
            st.error("❌ 로컬 DB가 없습니다. 먼저 python ingest.py를 실행해 주세요.")
        st.session_state.active_store = st.session_state.local_vector_store

    # [모드 B] 실시간 파일 업로드
    else:
        st.markdown(SIDEBAR_UPLOAD_HTML, unsafe_allow_html=True)
        
        # ⭐️ [핵심 보완] 잘려 보이던 가짜 HTML 태그를 지우고, 이 자리에 실제 기능하는 업로더를 삽입
        # label_visibility="collapsed" 옵션으로 지저분한 기본 타이틀 텍스트를 숨겨 카드와 일체화시킵니다.
        uploaded_file = st.file_uploader(
            "문서 파일 업로드",
            type=["pdf", "txt"],
            accept_multiple_files=False,
            label_visibility="collapsed"
        )
        
        # 새로운 파일이 감지되면 동적 임베딩 수행
        if uploaded_file is not None:
            if st.session_state.last_uploaded_name != uploaded_file.name:
                with st.spinner("문서를 실시간으로 분석하는 중... ✨"):
                    file_name = uploaded_file.name
                    docs = []
                    if file_name.endswith('.txt'):
                        text = uploaded_file.read().decode("utf-8")
                        docs.append(Document(page_content=text, metadata={"source": file_name}))
                    elif file_name.endswith('.pdf'):
                        pdf_reader = pypdf.PdfReader(uploaded_file)
                        text = ""
                        for page in pdf_reader.pages:
                            text += page.extract_text() or ""
                        docs.append(Document(page_content=text, metadata={"source": file_name}))

                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=20)
                    splits = text_splitter.split_documents(docs)

                    # 실시간 업로드는 Chroma를 메모리에서 사용
                    st.session_state.active_store = Chroma.from_documents(
                        documents=splits, embedding=embeddings_model
                    )
                    st.session_state.last_uploaded_name = uploaded_file.name
                    st.session_state.qa_predictions = None # 새 파일이므로 예측 초기화
                    st.session_state.selected_prediction_index = None
                    st.success("✅ 임시 파일 분석 완료!")
        else:
            st.session_state.active_store = None
            st.session_state.last_uploaded_name = None
            
    st.markdown(SIDEBAR_STEP2_HTML, unsafe_allow_html=True)

active_store = st.session_state.get('active_store')

# =========================================================================
# ⭐️ [핵심 추가] 활성 문서 기반으로 예상 Q&A 생성
# =========================================================================
if active_store and st.session_state.qa_predictions is None:
    with st.spinner("문서 내용을 기반으로 예상 질문과 답변을 생성하는 중... 🔮"):
        retriever = active_store.as_retriever(search_kwargs={"k": 5})
        docs = retriever.invoke("이 문서의 주요 내용에 대한 질문과 답변 쌍을 만들어줘.")
        context_text = "\n\n".join(doc.page_content for doc in docs)

        if context_text.strip():
            qa_gen_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
            qa_gen_prompt_template = """
            You are a helpful assistant who analyzes a given context and creates two relevant questions and answers based on it.
            Provide the output as a valid JSON list of objects, where each object has a "question" key and an "answer" key.
            Example: [{"question": "What is the main topic?", "answer": "The main topic is..."}, {"question": "Who is the author?", "answer": "The author is..."}]

            Context:
            {context}
            """
            qa_gen_prompt = PromptTemplate.from_template(qa_gen_prompt_template)
            qa_gen_chain = qa_gen_prompt | qa_gen_llm | SimpleJsonOutputParser()
            try:
                st.session_state.qa_predictions = qa_gen_chain.invoke({"context": context_text})
            except Exception:
                st.session_state.qa_predictions = [] # 파싱 오류 시 빈 리스트로 처리
        else:
            st.session_state.qa_predictions = []


# 메인 UI 레이아웃
st.markdown(f'<div class="page-card" style="border-left: 5px solid #1f4ed8; background: #f8faff; padding: 16px;"><span style="font-weight:700; color:#1f4ed8;">현재 활성화된 모드:</span> {chat_mode}</div>', unsafe_allow_html=True)

# ⭐️ [핵심 추가] 예상 질문 카드 UI
# ⭐️ [수정] 질문 목록이 실제로 있을 때만 UI를 그리도록 길이를 확인하는 방어 코드 추가
if st.session_state.get("qa_predictions") and len(st.session_state.qa_predictions) > 0:
    st.markdown(
        """
        <div class="page-card" style="margin-top: 1.5rem; padding-bottom: 16px;">
          <div class="section-label">SUGGESTED QUESTIONS</div>
          <div class="section-title">💡 AI가 추천하는 예상 질문</div>
          <p class="upload-hint">아래 카드를 클릭하면 미리 생성된 답변을 바로 확인할 수 있습니다.</p>
          <div style="margin-top: 1rem;"></div>
        """,
        unsafe_allow_html=True
    )
    cols = st.columns(len(st.session_state.qa_predictions))
    for i, qa in enumerate(st.session_state.qa_predictions):
        with cols[i]:
            is_selected_class = "prediction-card-selected" if str(i) == st.session_state.get("selected_prediction_index") else ""
            st.markdown(
                f"""
                <a href="?select_prediction={i}" target="_self" style="text-decoration: none;">
                    <div class="prediction-card {is_selected_class}">
                        <p class="prediction-question">{qa['question']}</p>
                    </div>
                </a>""", unsafe_allow_html=True
            )
    st.markdown("</div>", unsafe_allow_html=True) # page-card div 닫기

st.markdown(
    r"""
    <div class="page-card">
      <div class="section-label">CHAT</div>
      <div class="section-title">💬 문서 요정과 대화하기 (Multi-Query 작동 중)</div>
      <p class="upload-hint">질문을 입력하면 AI 비서가 다각도로 질문을 재해석하여 정확한 정보를 찾아냅니다.</p>
      <div class="panel-divider"></div>
    """,
    unsafe_allow_html=True,
)

if not st.session_state.messages and not st.session_state.get("qa_predictions"):
    if chat_mode == "📌 고정 문서 DB (chroma_db)":
        st.info("고정 문서 로드가 완료되었습니다. 아래 대화창에 바로 질문을 입력하세요!")
    else:
        st.info("왼쪽 사이드바 카드 영역에 PDF 또는 TXT 문서를 업로드해 주시면 대화가 시작됩니다.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

st.markdown(r"""<div class="panel-divider"></div></div>""", unsafe_allow_html=True)

# 사용자 질문 입력 및 핵심 로직 처리
if user_query := st.chat_input("선택한 문서 소스에 대해 궁금한 점을 물어보세요!"):
    # 사용자가 직접 질문을 입력하면, 선택된 예상 질문 상태는 초기화
    st.session_state.selected_prediction_index = None

    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("assistant"):
        if active_store is None:
            if chat_mode == "📌 고정 문서 DB (chroma_db)":
                response_text = "로컬 디스크에 생성된 `chroma_db` 폴더를 찾을 수 없습니다. `python ingest.py`를 먼저 실행해 주세요."
            else:
                response_text = "먼저 왼쪽 업로드 카드 영역에서 문서를 드래그하여 업로드해 주세요. 📂"
            st.markdown(response_text)
        else:
            with st.spinner("요정이 다각도로 질문을 분석하여 문서를 검색하고 있습니다... 💭"):
                # 기본 검색기 빌드
                retriever = active_store.as_retriever(search_kwargs={"k": 3})
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=512)

                # 제시해주신 프롬프트 가이드 반영
                system_prompt = (
                    "너는 질문-답변을 돕는 유능한 비서야. "
                    "아래 제공된 맥락(context)만을 사용하여 질문에 답해줘. "
                    "답을 모르면 모른다고 하고, 절대 답변을 지어내지 마.\n\n"
                    "Context:\n{context}"
                )
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{input}"),
                ])

                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)

                # 순수 LCEL 파이프라인 구조 유지로 에러 차단
                rag_chain = (
                    {"context": retriever | format_docs, "input": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )

                response_text = rag_chain.invoke(user_query)
                st.markdown(response_text)

    st.session_state.messages.append({"role": "assistant", "content": response_text})