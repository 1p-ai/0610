import os
import streamlit as st
import pypdf
import pandas as pd
import numpy as np
from langchain_core.documents import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="📄 다중 문서 융합 분석기 (PDF/TXT/CSV)", page_icon="📄", layout="wide")
st.title("📄 다중 파일 융합 분석 및 교차 Q&A 시스템")
st.markdown("---")

# --- 사이드바: API 키 입력 및 설정 ---
st.sidebar.title("⚙️ 설정 및 인증")

if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""

if st.sidebar.button("🔄 API 키 초기화"):
    st.session_state.openai_api_key = ""
    st.session_state.pdf_messages = []
    st.session_state.selected_card = None
    st.session_state.card_question = ""
    st.rerun()

user_key = st.sidebar.text_input(
    "OpenAI API Key를 입력하세요:", 
    value=st.session_state.openai_api_key, 
    type="password",
    help="오른쪽 버튼을 통해 발급 가이드를 확인하실 수 있습니다."
)

st.session_state.openai_api_key = user_key

if st.session_state.openai_api_key:
    st.sidebar.success("API 키가 입력되었습니다! 🎉")
else:
    st.sidebar.warning("앱을 사용하려면 OpenAI API Key를 입력해야 합니다.")
    st.sidebar.link_button(
        "💡 OpenAI API Key 발급방법 안내보기",
        "https://flextudio.com/blog/openai-1",
        help="클릭하면 API Key 발급 방법 안내 페이지가 새 창으로 열립니다."
    )


# --- 하이브리드 교차 컨텍스트 엔진 및 함수 정의 ---

@st.cache_resource(show_spinner="업로드된 모든 문서를 교차 인덱싱하여 통합 벡터 데이터로 변환 중입니다...")
def create_multi_retriever(_uploaded_files, _openai_api_key):
    """업로드된 여러 개의 파일들을 전수 조사하여, 하나의 통합된 가상 하이브리드 리트리버를 생성합니다."""
    if not _uploaded_files or not _openai_api_key:
        return None

    all_combined_texts = []
    combined_csv_summary = ""
    has_large_file = False

    for file in _uploaded_files:
        file_name = file.name
        file_docs = []

        try:
            if file_name.endswith('.pdf'):
                pdf_reader = pypdf.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""
                file_docs = [Document(page_content=text, metadata={"source": file_name})]

            elif file_name.endswith('.txt'):
                text = file.read().decode("utf-8")
                file_docs = [Document(page_content=text, metadata={"source": file_name})]

            elif file_name.endswith('.csv'):
                df = None
                encodings = ['utf-8', 'cp949', 'euc-kr']
                
                for enc in encodings:
                    try:
                        file.seek(0)
                        for skip in [15, 16, 17, 18, 14, 3, 1, 0]:
                            file.seek(0)
                            temp_df = pd.read_csv(file, encoding=enc, skiprows=skip)
                            temp_df.columns = temp_df.columns.str.strip()
                            if any(col in temp_df.columns for col in ["시군구", "단지명", "시군구별", "기본항목"]):
                                df = temp_df
                                break
                        if df is not None:
                            break
                    except Exception:
                        continue
                
                if df is None:
                    st.error(f"[{file_name}] CSV 파일의 구조나 인코딩을 분석할 수 없어 제외합니다.")
                    continue
                
                total_cells = len(df) * len(df.columns)
                
                file_rows_text = []
                for idx, row in df.iterrows():
                    row_items = []
                    for col, val in row.items():
                        if pd.notna(val):
                            row_items.append(f"{str(col).strip()}: {str(val).strip()}")
                    row_text = f"[파일: {file_name} / 행 {idx+1}] " + ", ".join(row_items)
                    file_rows_text.append(row_text)
                    file_docs.append(Document(page_content=row_text, metadata={"source": file_name, "row": idx}))
                
                if total_cells < 3000:
                    combined_csv_summary += (
                        f"\n[데이터 테이블 목록 - 파일명: {file_name}]\n"
                        f"컬럼 구조: {list(df.columns)}\n"
                        "전체 행 데이터:\n" + "\n".join(file_rows_text) + "\n"
                    )
                else:
                    has_large_file = True
                    preview_rows = [file_rows_text[i] for i in range(min(3, len(file_rows_text)))]
                    combined_csv_summary += (
                        f"\n[대용량 통계 테이블 요약 - 파일명: {file_name}]\n"
                        f"구조: {len(df)}행 x {len(df.columns)}열 (총 {total_cells}셀 데이터)\n"
                        f"컬럼 목록(상위 10개): {list(df.columns[:10])} ... 외 {len(df.columns)-10}개\n"
                        f"상위 샘플 데이터:\n" + "\n".join(preview_rows) + "\n"
                    )

            if file_name.endswith('.csv'):
                all_combined_texts.extend(file_docs)
            else:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=120)
                split_docs = text_splitter.split_documents(file_docs)
                all_combined_texts.extend(split_docs)

        except Exception as e:
            st.error(f"{file_name} 파일을 처리하는 중 오류 발생: {e}")
            continue

    if not all_combined_texts:
        return None

    try:
        embeddings = OpenAIEmbeddings(api_key=_openai_api_key)
        doc_texts = [doc.page_content for doc in all_combined_texts]
        doc_embeddings = embeddings.embed_documents(doc_texts)
        
        search_k = 3 if has_large_file else 6

        def multi_hybrid_search(query_text):
            query_vec = np.array(embeddings.embed_query(query_text))
            q_norm = np.linalg.norm(query_vec)
            
            scores = []
            for idx, doc_vec in enumerate(doc_embeddings):
                d_vec = np.array(doc_vec)
                d_norm = np.linalg.norm(d_vec)
                score = np.dot(query_vec, d_vec) / (q_norm * d_norm) if q_norm * d_norm > 0 else 0.0
                scores.append((score, all_combined_texts[idx]))
            
            scores.sort(key=lambda x: x[0], reverse=True)
            top_k_docs = [doc for score, doc in scores[:search_k]]
            
            semantic_context = "\n\n".join(
                f"[출처: {doc.metadata.get('source')}] {doc.page_content}" for doc in top_k_docs
            )
            
            final_context = ""
            if combined_csv_summary:
                final_context += f"=== 업로드된 구조화 데이터 가이드 테이블 ===\n{combined_csv_summary}\n=========================================\n\n"
            
            final_context += f"=== 실시간 의미 검색 교차 매칭 문맥 ===\n{semantic_context}"
            return final_context

        return RunnableLambda(multi_hybrid_search)

    except Exception as e:
        st.error(f"통합 임베딩 엔진 구축 실패: {e}")
        return None


# --- 메인 앱 로직 ---

if not st.session_state.openai_api_key:
    st.info("👈 왼쪽 사이드바에 OpenAI API Key를 먼저 입력해주세요!")
    st.stop()

openai_api_key = st.session_state.openai_api_key

uploaded_files = st.file_uploader(
    "분석할 파일들을 업로드하세요. (최대 3개 파일까지 동시 업로드 및 융합 분석 지원)", 
    type=["pdf", "txt", "csv"],
    accept_multiple_files=True
)

if uploaded_files and len(uploaded_files) > 3:
    st.error(f"🚨 **분석 불가 알림**: 현재 시스템의 안정적인 분석 지원 범위를 초과했습니다. (업로드된 파일 수: {len(uploaded_files)}개 / 최대 지원 한도: 3개)")
    st.stop()

# 세션 상태 변수 안전 초기화
if "pdf_messages" not in st.session_state:
    st.session_state.pdf_messages = []
if "selected_card" not in st.session_state:
    st.session_state.selected_card = None
if "card_question" not in st.session_state:
    st.session_state.card_question = ""

# 파일 변경 시 세션 변수 완벽 초기화
if "last_uploaded_files" not in st.session_state or st.session_state.last_uploaded_files != uploaded_files:
    st.session_state.last_uploaded_files = uploaded_files
    st.session_state.pdf_messages = []
    st.session_state.selected_card = None
    st.session_state.card_question = ""

if uploaded_files:
    retriever = create_multi_retriever(uploaded_files, openai_api_key)

    if retriever:
        # --- 추천 질문 카드 UI 구현부 ---
        st.markdown("### 💡 파일 맞춤형 추천 질문 카드가 생성되었습니다.")
        st.markdown("<small>원하는 예시 질문 카드를 클릭하시면 자동으로 분석이 수행됩니다.</small>", unsafe_allow_html=True)
        
        has_csv = any(f.name.endswith('.csv') for f in uploaded_files)
        if has_csv:
            q1 = "2024년 3월 서울 종로구 명륜2가에 있는 아남1 아파트의 건축년도와 도로명 주소 정보를 알려주세요."
            q2 = "명륜2가 아남1 아파트의 거래 내역 중에서, 거래금액이 125,000만 원 이상이거나 3월 25일 이후에 계약된 건의 층수는 각각 몇 층인가요?"
            q3 = "업로드된 출생 통계 데이터(인구수 트렌드)와 아파트 실거래가 데이터를 결합하여 부동산 수요 변화 관점으로 융합 리포트를 요약해 주세요."
        else:
            q1 = "업로드된 전체 문서 내용의 핵심 요약본을 작성해 주세요."
            q2 = "이 문서에서 가장 중요하게 다루고 있는 핵심 키워드 3개와 그 이유를 분석해 주세요."
            q3 = "문서의 정보와 팩트를 기반으로 실무에서 활용할 수 있는 Q&A 가이드를 작성해 주세요."

        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.session_state.selected_card == 1:
                st.success(f"**선택됨 (과제 1형)**\n\n{q1}")
            else:
                st.info(f"**추천 질문 1**\n\n{q1}")
            if st.button("질문 1 실행 🚀", key="btn_q1"):
                st.session_state.selected_card = 1
                st.session_state.card_question = q1
                st.rerun()

        with col2:
            if st.session_state.selected_card == 2:
                st.success(f"**선택됨 (과제 2형)**\n\n{q2}")
            else:
                st.info(f"**추천 질문 2**\n\n{q2}")
            if st.button("질문 2 실행 🚀", key="btn_q2"):
                st.session_state.selected_card = 2
                st.session_state.card_question = q2
                st.rerun()

        with col3:
            if st.session_state.selected_card == 3:
                st.success(f"**선택됨 (융합 분석형)**\n\n{q3}")
            else:
                st.info(f"**추천 질문 3**\n\n{q3}")
            if st.button("질문 3 실행 🚀", key="btn_q3"):
                st.session_state.selected_card = 3
                st.session_state.card_question = q3
                st.rerun()

        st.markdown("---")

        # --- [에러 해결 포인트 1] 입력창 및 카드 주입 이벤트 동시 접수 처리 ---
        active_question = ""
        chat_input_val = st.chat_input("질문을 입력하거나 상단의 추천 카드를 눌러보세요!")
        
        if st.session_state.card_question:
            active_question = st.session_state.card_question
            st.session_state.card_question = ""  # 소모 후 초기화
        elif chat_input_val:
            active_question = chat_input_val
            st.session_state.selected_card = None  # 직접 타이핑 시 카드 하이라이트 해제

        # 질문이 들어온 경우 세션 메시지에 먼저 안전하게 추가(Append)를 완결시킵니다.
        if active_question:
            st.session_state.pdf_messages.append({"role": "user", "content": active_question})

        # --- [에러 해결 포인트 2] 대화 기록 출력 (딕셔너리 .get() 구조 방어 적용) ---
        for message in st.session_state.pdf_messages:
            msg_role = message.get("role")
            msg_content = message.get("content", "")
            if msg_role:  # role 키가 정상적으로 존재하는 메시지만 안전하게 렌더링
                with st.chat_message(msg_role):
                    st.markdown(msg_content)

        # --- RAG 응답 스트리밍 구동부 ---
        if active_question:
            with st.chat_message("assistant"):
                llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0,  
                    streaming=True,
                    api_key=openai_api_key,
                )

                prompt = ChatPromptTemplate.from_template(
                    "당신은 제공된 문서를 바탕으로 통계 분석, 필터링, 수치 비교를 수행하는 전문 데이터 분석가 AI입니다.\n\n"
                    "[지침 사항]\n"
                    "1. 주어진 문맥(Context)에 기재된 정보만을 정밀하게 분석하여 질문에 답하세요.\n"
                    "2. 수치 비교(예: 최고/최저 금액 산출), 날짜 필터링(예: 특정 날짜 이후 계약 건 추출) 요건이 있을 경우, "
                    "문맥에 열거된 모든 행 데이터를 꼼꼼히 대조하여 누락 없이 논리적이고 정확하게 계산해 답변하세요.\n"
                    "3. 부동산 정보와 관련 없는 이기종 파일(예: 출생아 수 통계)이 주어지더라도, 해당 데이터가 가리키는 "
                    "지역(시군구) 정보와 트렌드를 부동산 시장 활성화나 배후 수요 관점 등과 연결지어 창의적이면서도 논리적인 부동산 융합 분석 리포트를 작성해 주세요.\n"
                    "4. 추측이나 왜곡 없이 문맥에 있는 팩트만 사용하세요.\n\n"
                    "Context:\n{context}\n\n"
                    "Question: {input}\n\n"
                    "정밀 분석 답변:"
                )

                rag_chain = (
                    {"context": retriever, "input": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )

                def response_generator():
                    for chunk in rag_chain.stream(active_question):
                        yield chunk

                response = st.write_stream(response_generator())
                st.session_state.pdf_messages.append({"role": "assistant", "content": response})
else:
    st.info("💡 **사용법**: 분석기 화면에 분석하고자 하는 파일(PDF/TXT/CSV)을 업로드하면 맞춤형 자동 추천 질문 카드가 활성화됩니다.")