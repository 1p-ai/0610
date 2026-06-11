
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
import tempfile
import pypdf

# 문서 분할기
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector DB
from langchain_chroma import Chroma

# LangChain 모델 및 코어
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI


# LangChain Expression Language (LCEL) 및 파서
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.title("PDF File Reader")
st.write("---")

# PDF 업로드 영역
uploaded_file = st.file_uploader( "PDF 파일을 업로드하세요",  type=["pdf"] )
st.write("---")


# ==========================================================
# PDF → Document 변환 함수
# ==========================================================
def pdf_to_documents(uploaded_file):
    """
    업로드된 PDF 파일을 메모리에서 직접 읽어 pypdf를 사용해
    LangChain Document 리스트(페이지별) 형태로 변환합니다.

    처리 과정:
    1. pypdf로 업로드된 파일의 바이트 스트림을 직접 읽기
    2. 페이지별로 텍스트를 추출하여 Document 객체 생성

    return:
        pages (list[langchain_core.documents.Document])
    """
    pdf_reader = pypdf.PdfReader(uploaded_file)
    docs = []
    for i, page in enumerate(pdf_reader.pages):
        text = page.extract_text()
        if text:
            docs.append(Document(page_content=text, metadata={'source': uploaded_file.name, 'page': i + 1}))
    return docs


if uploaded_file is not None:
    pages = pdf_to_documents(uploaded_file)
    st.success(  f"PDF 로딩 완료 : {len(pages)} 페이지"   )

    text_splitter = RecursiveCharacterTextSplitter(
        # 한 조각의 최대 글자 수
        chunk_size=1000,

        # 앞뒤 중복 문자, 문맥 유지를 위해 사용
        chunk_overlap=200
    )


    # Document 분할
    texts = text_splitter.split_documents(  pages   )

    st.info(  f"문서 조각 개수 : {len(texts)}"   )

    # -------------------------------
    # 3. Embedding 생성
    # -------------------------------
    embeddings = OpenAIEmbeddings()
    # -------------------------------
    # 4. Vector Database 생성
    # -------------------------------
    db = Chroma.from_documents(  documents=texts,  embedding=embeddings   )

    # -------------------------------
    # 5. Retriever 생성
    # -------------------------------
    retriever = db.as_retriever(  search_kwargs={   "k": 3   }   )

    # ======================================================
    # 질문 입력
    # ======================================================
    st.header( "PDF에게 질문하세요"    )
    question = st.text_input( "질문 입력"   )

    if st.button("질문하기"):
        if question.strip()=="":
            st.warning(  "질문을 입력해주세요"     )

        else:
            with st.spinner( "AI가 답변 생성중..."   ):

                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

                prompt = ChatPromptTemplate.from_template(
                    """
                    당신은 PDF 문서 분석 전문가입니다.

                    아래 Context 내용을 참고하여
                    질문에 답변하세요.

                    Context: {context}
                    Question: {input}
                    """
                )

                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)

                rag_chain = (
                    {"context": retriever | format_docs, "input": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )
                answer = rag_chain.invoke(question)
                st.write(answer)