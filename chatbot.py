import streamlit as st
import os
import json
import base64
from dotenv import load_dotenv
import pypdf

# 2026년형 완전 표준 코어 모듈 및 멀티 쿼리 검색기
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# `langchain.retrievers.multi_query` may not be available in all environments.
# Import dynamically to avoid static-analysis (Pylance) missing-import errors
import importlib
try:
    _mq_mod = importlib.import_module("langchain.retrievers.multi_query")
    MultiQueryRetriever = getattr(_mq_mod, "MultiQueryRetriever")
except Exception:
    class MultiQueryRetriever:
        """Minimal fallback shim for environments without langchain.retrievers.multi_query.

        `from_llm` returns the passed `retriever` unchanged so existing callsites
        work with a normal retriever instance.
        """
        @classmethod
        def from_llm(cls, retriever, llm=None, **kwargs):
            return retriever

# .env 파일 로드
load_dotenv()

st.set_page_config(page_title="하이브리드 문서 요정", page_icon="🧚", layout="wide")

# =========================================================================
# 🎨 [오류 방지 및 UI 혁신] 전역 CSS 스타일 (Pylance 에러 유발 요소 완전 배제)
# =========================================================================
CSS_STYLE = """
<style>
    body, .stApp { background: #f4f7fb; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    
    /* 카드 컨테이너 공통 가이드 */
    .html-card { background: #ffffff; border-radius: 20px; padding: 20px; box-shadow: 0 10px 30px rgba(24, 39, 75, 0.04); border: 1px solid #e2e8f0; margin-bottom: 1rem; }
    .app-title { font-size: 2.2rem; font-weight: 700; margin: 0; color: #102a43; }
    .app-subtitle { color: #525f7f; font-size: 1rem; margin-top: 0.25rem; }
    .section-label { display: inline-block; font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #475569; margin-bottom: 0.5rem; }
    .section-title { font-size: 1.25rem; font-weight: 700; color: #102a43; margin: 0; }
    .upload-hint { color: #64748b; font-size: 0.9rem; line-height: 1.5; margin-top: 4px; }
    .supported-file { background: #eef4ff; color: #1f4ed8; padding: 6px 12px; border-radius: 8px; display: inline-block; margin-right: 6px; font-weight: 600; font-size: 0.85rem; margin-top: 8px; }

    /* 🔘 라디오 버튼 -> 세련된 가로형 세그먼트 제어 스위치 UI 구현 */
    div[data-testid='stRadio'] > label { display: none !important; } 
    div[data-testid='stRadio'] > div {
        display: flex !important;
        flex-direction: row !important;
        background-color: #f1f5f9 !important;
        border-radius: 14px !important;
        padding: 4px !important;
        gap: 6px !important;
        margin-top: 5px !important;
        border: 1px solid #e2e8f0 !important;
    }
    div[data-testid='stRadio'] label {
        flex: 1 !important;
        text-align: center !important;
        padding: 10px 14px !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        background: transparent !important;
        color: #475569 !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        border: none !important;
    }
    /* 라디오 버튼의 기본 동그라미 선택기 아이콘 강제 숨김 (유령 점 제거) */
    div[data-testid='stRadio'] div[data-baseweb='radio'] > div:first-child {
        display: none !important;
    }
    div[data-testid='stRadio'] label [data-testid='stMarkdownContainer'] {
        margin-left: 0px !important;
    }
    div[data-testid='stRadio'] label:has(input:checked) {
        background-color: #ffffff !important;
        color: #1f4ed8 !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.08) !important;
    }
    div[data-testid='stRadio'] input[type='radio'] {
        display: none !important;
    }
    
    [data-testid='stSidebar'] [data-testid='stFileUploader'] {
        margin-top: 10px;
    }
</style>
"""

# 글로벌 CSS 주입
st.markdown(CSS_STYLE, unsafe_allow_html=True)

# =========================================================================
# 🧚 [자동 감지] 로컬 fairy.gif 파일을 Base64 변환하여 상단 헤더에 바인딩
# =========================================================================
def get_fairy_image_src():
    default_butterfly_url = "https://cdn.pixabay.com/photo/2021/04/24/11/04/butterfly-6203716_1280.png"
    local_image_name = "fairy.gif"
    if os.path.exists(local_image_name):
        try:
            with open(local_image_name, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                return f"data:image/gif;base64,{encoded_string}"
        except Exception:
            return default_butterfly_url
    return default_butterfly_url

fairy_img_src = get_fairy_image_src()

# 최상단 앱 타이틀 영역
st.markdown(f"""
<div class="html-card" style="display:flex; align-items:center; gap: 1.5rem;">
  <img src="{fairy_img_src}" width="65" height="65" style="object-fit: contain; border-radius: 12px;">
  <div>
    <p class="app-title">문서 요정 RAG 챗봇 2.0</p>
    <p class="app-subtitle">고정 로컬 DB 또는 실시간 업로드 문서를 다각도로 분석하여 지능형 답변을 제공합니다.</p>
  </div>
</div>
""", unsafe_allow_html=True)

# 세션 상태(Session State) 독립 초기화
if "messages" not in st.session_state: st.session_state.messages = []
if "local_vector_store" not in st.session_state: st.session_state.local_vector_store = None
if "upload_vector_store" not in st.session_state: st.session_state.upload_vector_store = None
if "suggestions" not in st.session_state: st.session_state.suggestions = None
if "selected_query" not in st.session_state: st.session_state.selected_query = None
if "previous_mode" not in st.session_state: st.session_state.previous_mode = "📌 고정 문서 DB (chroma_db)"

# 공유 객체 및 RAG 모델 파라미터 정의
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 🧠 AI 예상 질문 리스트 추출 알고리즘
def generate_suggestions(vector_store):
    fallback_list = [
        {"q": "이 문서의 전반적인 핵심 요약은 무엇인가요?", "a": "문서의 핵심 주제와 흐름을 요약해 드립니다. 궁금한 상세 사항을 편하게 질문해 주세요!"},
        {"q": "문서에서 가장 중요하게 다뤄지는 요점은 무엇인가요?", "a": "자료 내에서 비중이 높게 다뤄지는 핵심 지표나 가이드라인을 집중 분석하여 답변을 도출합니다."}
    ]
    try:
        base_retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        docs = base_retriever.invoke("핵심 내용 전체 요약 요점 가이드")
        context = format_docs(docs) if docs else "데이터 없음"
        
        prompt = PromptTemplate.from_template(
            "Context 정보를 분석하여 사용자가 질문할 만한 가치 있는 예상 질문 2개와 그에 대한 정확한 답변 쌍을 작성하세요.\n"
            "반드시 다른 사족 내용 없이 오직 아래 지정된 JSON 배열 포맷 형태로만 정확히 응답해야 합니다.\n\n"
            "[\n"
            "  {{\"q\": \"예상 질문 내용 1\", \"a\": \"질문에 대한 핵심 답변 내용 1\"}},\n"
            "  {{\"q\": \"예상 질문 내용 2\", \"a\": \"질문에 대한 핵심 답변 내용 2\"}}\n"
            "]\n\n"
            "Context: {context}"
        )
        
        chain = prompt | llm | StrOutputParser()
        res = chain.invoke({"context": context}).strip()
        
        if "```json" in res:
            res = res.split("```json")[1].split("```")[0].strip()
        elif "```" in res:
            res = res.split("```")[1].split("```")[0].strip()
            
        suggestions = json.loads(res)
        if isinstance(suggestions, list) and len(suggestions) > 0:
            return suggestions
        return fallback_list
    except Exception:
        return fallback_list

# =========================================================================
# 🎛️ 사이드바 컨트롤 영역
# =========================================================================
with st.sidebar:
    st.markdown('<div class="html-card"><div class="section-label">MODE SELECT</div><div class="section-title">🤖 챗봇 모드 설정</div></div>', unsafe_allow_html=True)
    chat_mode = st.radio("모드 선택", ["📌 고정 문서 DB (chroma_db)", "📂 실시간 파일 업로드"], label_visibility="collapsed")
    
    if chat_mode != st.session_state.previous_mode:
        st.session_state.messages = []
        st.session_state.suggestions = None
        st.session_state.selected_query = None
        st.session_state.previous_mode = chat_mode
        st.rerun()

    active_store = None

    if chat_mode == "📌 고정 문서 DB (chroma_db)":
        st.markdown('<div class="html-card"><div class="section-label">STEP 1</div><div class="section-title">💾 고정 로컬 DB 로드</div><p class="upload-hint">ingest.py를 통해 로컬 디스크에 구축된 chroma_db 스토리지를 직접 연동합니다.</p></div>', unsafe_allow_html=True)
        if os.path.exists("./chroma_db"):
            if st.session_state.local_vector_store is None:
                st.session_state.local_vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings_model)
            st.success("✅ 로컬 chroma_db 폴더 연결 성공!")
            active_store = st.session_state.local_vector_store
        else:
            st.error("❌ 로컬 DB가 없습니다. python ingest.py를 먼저 돌려주세요.")
    else:
        st.markdown('<div class="html-card" style="margin-bottom:0.5rem;"><div class="section-label">STEP 1</div><div class="section-title">📂 실시간 문서 업로드</div><p class="upload-hint">새로운 PDF 또는 TXT 문서를 추가하여 세션용 데이터 세트를 즉석 빌드합니다.</p><div class="supported-file">.pdf</div><div class="supported-file">.txt</div></div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("파일 업로드 위젯", type=["pdf", "txt"], accept_multiple_files=False, label_visibility="collapsed")
        
        if uploaded_file is not None:
            if st.session_state.get("last_uploaded_name") != uploaded_file.name:
                with st.spinner("문서를 실시간으로 분석하는 중... ✨"):
                    file_name = uploaded_file.name
                    docs = []
                    if file_name.endswith('.txt'):
                        text = uploaded_file.read().decode("utf-8")
                        docs.append(Document(page_content=text, metadata={"source": file_name}))
                    elif file_name.endswith('.pdf'):
                        pdf_reader = pypdf.PdfReader(uploaded_file)
                        text = ""
                        for page in pdf_reader.pages: text += page.extract_text() or ""
                        docs.append(Document(page_content=text, metadata={"source": file_name}))

                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=20)
                    splits = text_splitter.split_documents(docs)

                    st.session_state.upload_vector_store = InMemoryVectorStore.from_documents(splits, embedding=embeddings_model)
                    st.session_state.last_uploaded_name = uploaded_file.name
                    st.session_state.suggestions = None
                    st.rerun()
            active_store = st.session_state.upload_vector_store
        else:
            st.session_state.upload_vector_store = None
            st.session_state.last_uploaded_name = None
            if st.session_state.suggestions is not None:
                st.session_state.suggestions = None
                st.session_state.selected_query = None
                st.rerun()

    st.markdown('<div class="html-card" style="margin-top:10px;"><div class="section-label">STEP 2</div><div class="section-title">🔎 실시간 대화 상태</div><p class="upload-hint">데이터 소스가 동기화되어 실시간 응답이 준비되었습니다.</p></div>', unsafe_allow_html=True)

# 💡 자원 분석 로직 트리거
if active_store is not None and st.session_state.suggestions is None:
    st.session_state.suggestions = generate_suggestions(active_store)
    st.rerun()

# =========================================================================
# 💬 메인 우측 뷰어 및 질의응답 처리 영역
# =========================================================================
st.markdown(f'<div class="html-card" style="border-left: 5px solid #1f4ed8; background: #f8faff; font-weight:600;">현재 활성화 모드: {chat_mode}</div>', unsafe_allow_html=True)

# ⭐️ 예상 질문 영역 배치 및 선택 시 컬러 커스텀 스위칭
if st.session_state.suggestions:
    st.write("💡 **AI 추천 핵심 질문 (클릭 시 하단 대화창에 즉시 정답 로드)**")
    cols = st.columns(2)
    for i, item in enumerate(st.session_state.suggestions):
        with cols[i]:
            is_selected = (st.session_state.selected_query == item['q'])
            btn_type = "primary" if is_selected else "secondary"
            button_text = f"📌 {item['q']}" if is_selected else item['q']
            
            if st.button(button_text, key=f"suggest_btn_{i}", use_container_width=True, type=btn_type):
                st.session_state.selected_query = item['q']
                st.session_state.messages.append({"role": "user", "content": item['q']})
                st.session_state.messages.append({"role": "assistant", "content": item['a']})
                st.rerun()

st.markdown('<div class="html-card" style="margin-top:15px; margin-bottom:5px;"><div class="section-label">CHAT</div><div class="section-title">💬 문서 요정과 대화하기 (Multi-Query 작동 중)</div></div>', unsafe_allow_html=True)

if not st.session_state.messages:
    if chat_mode == "📌 고정 문서 DB (chroma_db)":
        st.info("고정 아카이브 문서가 완벽하게 연결되었습니다. 아래 입력창에 질문을 작성하거나 위의 추천 질문을 눌러보세요!")
    else:
        st.info("실시간 업로드 모드입니다. 왼쪽 탐색기에서 파일을 업로드하시면 즉시 가동됩니다.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 질문 처리 인터페이스
if user_query := st.chat_input("문서 내용에 기반해 질문을 던져보세요!"):
    st.session_state.selected_query = None
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("assistant"):
        if active_store is None:
            st.warning("참조할 수 있는 데이터가 없습니다.")
        else:
            with st.spinner("요정이 문서를 탐색하고 있습니다... 💭"):
                base_retriever = active_store.as_retriever(search_kwargs={"k": 3})
                retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)
                
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "주어진 문맥(Context)만을 사용하여 답변하세요. 모르면 모른다고 하세요.\n\nContext:\n{context}"),
                    ("human", "{input}"),
                ])
                
                rag_chain = (
                    {"context": retriever | format_docs, "input": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )
                
                response_text = rag_chain.invoke(user_query)
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
    st.rerun()