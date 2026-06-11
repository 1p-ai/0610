import os
import streamlit as st
from dotenv import load_dotenv
import pypdf
from langchain_core.documents import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# .env 파일에서 환경 변수 로드
load_dotenv()

st.set_page_config(page_title="📄 PDF 파일 분석기", page_icon="📄", layout="wide")
st.title("📄 PDF 파일 분석 및 Q&A")
st.markdown("---")

# --- 함수 및 클래스 정의 ---

@st.cache_resource(show_spinner="PDF를 분석하여 벡터 데이터로 변환 중입니다...")
def create_retriever(_uploaded_file, _openai_api_key):
    """업로드된 PDF 파일을 처리하여 LangChain 리트리버를 생성하고 캐시에 저장합니다."""
    if not _uploaded_file:
        return None

    # pypdf를 사용하여 메모리에서 직접 PDF 처리
    pdf_reader = pypdf.PdfReader(_uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""

    # Document 객체 생성
    docs = [Document(page_content=text, metadata={"source": _uploaded_file.name})]

    # 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_documents(docs)

    # 임베딩 및 벡터 DB 생성 (Chroma)
    embeddings = OpenAIEmbeddings(api_key=_openai_api_key)
    db = Chroma.from_documents(documents=texts, embedding=embeddings)

    # 리트리버 생성
    retriever = db.as_retriever(search_kwargs={"k": 3})
    return retriever

def format_docs(docs):
    """검색된 문서들을 프롬프트에 맞게 포맷합니다."""
    return "\n\n".join(doc.page_content for doc in docs)

# --- 메인 앱 로직 ---

# 1. API 키 확인
if "OPENAI_API_KEY" not in os.environ:
    st.error("API 키가 없습니다. .env 파일에 OPENAI_API_KEY를 설정해주세요.")
    st.stop()
openai_api_key = os.environ["OPENAI_API_KEY"]

# 2. 파일 업로더
uploaded_file = st.file_uploader("분석할 PDF 파일을 업로드하세요.", type=["pdf"])

# 3. 세션 상태 초기화
if "pdf_messages" not in st.session_state:
    st.session_state.pdf_messages = []

# 파일이 업로드되지 않았거나, 다른 파일로 교체되면 대화 기록 초기화
if "last_uploaded_file" not in st.session_state or st.session_state.last_uploaded_file != uploaded_file:
    st.session_state.last_uploaded_file = uploaded_file
    st.session_state.pdf_messages = []

if uploaded_file:
    # 4. 리트리버 생성 (캐시된 결과 사용)
    retriever = create_retriever(uploaded_file, openai_api_key)

    # 5. 대화 기록 표시
    for message in st.session_state.pdf_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 6. 사용자 질문 입력
    if question := st.chat_input("PDF 내용에 대해 질문해보세요!"):
        st.session_state.pdf_messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # 7. RAG 체인 실행 및 답변 생성
        with st.chat_message("assistant"):
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
                streaming=True,
                api_key=openai_api_key,
            )

            prompt = ChatPromptTemplate.from_template(
                "당신은 PDF 문서에 대해 답변하는 AI 어시스턴트입니다. 주어진 문맥(Context)만을 사용하여 질문에 답변해주세요.\n\n"
                "Context:\n{context}\n\n"
                "Question: {input}\n\n"
                "답변:"
            )

            # LCEL을 사용한 RAG 체인 구성
            rag_chain = (
                {"context": retriever | format_docs, "input": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            # 스트리밍 방식으로 답변 출력
            response = st.write_stream(rag_chain.stream(question))
            st.session_state.pdf_messages.append({"role": "assistant", "content": response})
else:
    st.info("PDF 파일을 업로드하면 분석 및 질문이 시작됩니다.")
