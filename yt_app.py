import streamlit as st
import os
from dotenv import load_dotenv

from yt_ingest import ingest_youtube_video
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

st.set_page_config(
    page_title="YouSumm: 유튜브 요약 & Q&A 챗봇",
    page_icon="🎬",
    layout="wide"
)

# 세션 상태 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_video" not in st.session_state:
    st.session_state.processed_video = None
if "input_url" not in st.session_state:
    st.session_state.input_url = ""
if "sample_error" not in st.session_state:
    st.session_state.sample_error = False
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {} 
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = "기본 대화방"

# 공통 RAG 추론 함수
def run_rag_inference(question_text):
    try:
        persist_directory = "./chroma_db"
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        
        current_source = st.session_state.processed_video['url']
        current_title = st.session_state.processed_video['title']
        
        if st.session_state.processed_video['type'] == "youtube":
            retriever = db.as_retriever(search_kwargs={"k": 4, "filter": {"source": current_source}})
        else:
            retriever = db.as_retriever(search_kwargs={"k": 4, "filter": {"title": current_title}})
        
        template = """당신은 오직 현재 선택된 영상 콘텐츠의 자막 내용만을 기반으로 답변하는 전문 요약 비서입니다.
        제공된 문맥(Context)만을 사용하여 사용자의 질문에 정확하고 친절하게 답하세요.
        오직 현재 영상의 텍스트 범주 내에서만 답변하고, 관련 내용이 없다면 정중히 모른다고 하세요.

        [Context]
        {context}

        [Question]
        {question}

        [Answer] :
        """
        prompt = ChatPromptTemplate.from_template(template)
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt | llm | StrOutputParser()
        )
        return rag_chain.invoke(question_text)
    except Exception as e:
        return f"❌ 답변 생성 오류: {str(e)}"


# 메인 타이틀
st.title("🎬 YouSumm: YouTube 요약 및 Q&A 시스템")
st.markdown("유튜브 영상 링크나 로컬 파일을 분석하여 인공지능이 답변해 드립니다.")
st.divider()

# 프리미엄 팝업 모달창 정의
@st.dialog("👑 프리미엄 기능 제한 안내")
def show_premium_modal():
    st.markdown("### 🚀 더 강력한 AI 분석을 경험하세요!")
    st.write("자막이 없는 영상의 **음성 인식(STT)** 및 영상 화면을 직접 분석하는 **멀티모달 AI 기능**은 프리미엄 회원 전용입니다.")
    st.info("💡 **연간 회원권 가입 안내**\n\n아래 계좌로 **연간 100,000원**을 입금하시면 프리미엄 기능이 즉시 활성화됩니다.\n\n* 계좌번호: 신한은행 123-456-789012 (예금주: 주식회사 유썸)")
    
    # "확인했습니다" 버튼 클릭 시 작동하는 이벤트
    if st.button("확인했습니다", type="primary", use_container_width=True):
        # 1. 화면에 "뻥입니다!" 축하 토스트/성공 알림 띄우기
        st.toast("🎉 뻥입니다! 현재는 없는 기능입니다. 🤣", icon="🔥")
        
        # 2. 폭죽을 대신할 화려한 화면 축하 애니메이션 트리거 (Streamlit 내장 효과)
        st.balloons()
        
        # 3. 약간의 시간차를 두고 모달을 닫고 무료 모드로 리셋하기 위한 세션 제어
        # 사용자가 효과를 충분히 감상할 수 있도록 세션을 초기화하며 화면을 리프레시합니다.
        st.success("🎉 뻥입니다. 현재는 없는 기능입니다.")
        
        # 잠시 대기 후 리프레시를 원하시면 시간차를 주거나 바로 리런합니다.
        # 여기서는 즉시 새로고침하여 무료 모드로 안전하게 복귀시킵니다.
        st.session_state.app_mode_key = "일반 무료 모드 (자막 기반)" 
        st.container()


# ==================================================================
# 🏢 [TOP 영역] 설정 및 대상 입력 (상단 스플릿)
# ==================================================================
st.markdown("### ⚙️ 상단 영역: 멤버십 설정 및 분석 대상 지정")

# 가로 공간 확보를 위해 입력 설정판을 2분할 배치
top_col1, top_col2 = st.columns([1, 1], gap="medium")

with top_col1:
    st.caption("💎 요금제 토글")
    app_mode = st.radio(
        "요금제를 선택하세요:",
        ["일반 무료 모드 (자막 기반)", "👑 프리미엄 모드 (STT & 멀티모달)"],
        horizontal=True,
        label_visibility="collapsed"
    )
    if app_mode == "👑 프리미엄 모드 (STT & 멀티모달)":
        show_premium_modal()

with top_col2:
    st.caption("📂 소스 주입 방식")
    input_tab1, input_tab2 = st.tabs(["📺 유튜브 링크 입력", "📁 로컬 파일 업로드"])
    
    with input_tab1:
        sample_url = "https://www.youtube.com/watch?v=O5xeyoRL95U"
        
        def run_onestop_sample():
            st.session_state.input_url = sample_url
            st.session_state.yt_url_input = sample_url
            st.session_state.sample_error = False
            
            video_title = ingest_youtube_video(sample_url)
            if video_title:
                st.session_state.processed_video = {
                    "url": sample_url, 
                    "title": video_title, 
                    "type": "youtube"
                }
            else:
                st.session_state.sample_error = True

        st.button(
            "🔥 원스톱 빠른 예시보기 (클릭 시 자동 분석 및 로드)", 
            type="secondary", 
            on_click=run_onestop_sample
        )
        
        if st.session_state.sample_error:
            st.error("자막 추출에 실패했습니다.")
            
        youtube_url = st.text_input(
            "유튜브 URL 입력:",
            value=st.session_state.get("input_url", ""),
            placeholder="https://www.youtube.com/watch?v=...",
            key="yt_url_input",
            label_visibility="collapsed"
        )
        
        if st.button("유튜브 자막 분석하기", type="primary", key="yt_btn"):
            if not youtube_url:
                st.warning("유튜브 URL을 입력해 주세요.")
            else:
                with st.spinner("자막 추출 중..."):
                    st.session_state.sample_error = False
                    video_title = ingest_youtube_video(youtube_url)
                    if video_title:
                        st.success(f"🎉 분석 완료: {video_title}")
                        st.session_state.processed_video = {"url": youtube_url, "title": video_title, "type": "youtube"}
                    else:
                        st.session_state.sample_error = True
                        st.rerun()

    with input_tab2:
        uploaded_file = st.file_uploader(
            "파일 업로드 (MP4, MP3, WAV 등)",
            type=["mp4", "mp3", "wav"],
            label_visibility="collapsed"
        )
        if st.button("업로드 파일 분석하기", type="primary", key="file_btn"):
            if not uploaded_file:
                st.warning("파일을 먼저 선택해 주세요.")
            else:
                st.info(f"📁 파일 접수 완료: {uploaded_file.name}")
                with st.spinner("파일 벡터 DB 생성 중..."):
                    st.success("🎉 파일 분석 완료 (Mock 세팅)")
                    st.session_state.processed_video = {"url": None, "title": uploaded_file.name, "type": "file"}

st.divider()


# ==================================================================
# 💬 [BOTTOM 영역] 비디오 플레이어 + 대화방 일체화 (하단 스플릿)
# ==================================================================
st.markdown("### 💬 하단 영역: 현재 분석 중인 동영상 콘텐츠 및 Q&A 챗봇")

# 하단을 반반 균등 분할하여 왼쪽은 비디오 플레이어, 오른쪽은 채팅창 배치
bot_col1, bot_col2 = st.columns([1, 1], gap="large")

# ------------------------------------------------------------------
# 하단 왼쪽: 현재 분석 중인 콘텐츠 영역 (이동 완료)
# ------------------------------------------------------------------
with bot_col1:
    if st.session_state.processed_video:
        st.info(f"🎬 **현재 분석 중:** {st.session_state.processed_video['title']}")
        
        if st.session_state.processed_video['type'] == "youtube":
            st.video(st.session_state.processed_video['url'])
        else:
            st.warning("💡 로컬 파일의 경우 영상 플레이어는 프리미엄 결제 후 활성화됩니다.")

        st.markdown("💡 **대화 원스톱 팁:** 아래 버튼을 누르면 질문과 답변이 즉시 우측 채팅창에 반영됩니다!")
        
        tip_q1 = "이 영상의 핵심 내용을 3줄로 요약해줘."
        tip_q2 = "타임라인별 요점 리스트를 정리해줘."
        
        if st.button(f"💬 {tip_q1}", use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "content": tip_q1})
            with st.spinner("답변 작성 중..."):
                ans = run_rag_inference(tip_q1)
                st.session_state.chat_history.append({"role": "assistant", "content": ans})
            st.rerun()
                
        if st.button(f"💬 {tip_q2}", use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "content": tip_q2})
            with st.spinner("답변 작성 중..."):
                ans = run_rag_inference(tip_q2)
                st.session_state.chat_history.append({"role": "assistant", "content": ans})
            st.rerun()
    else:
        st.light_boxes = st.info("ℹ️ 상단 영역에서 유튜브 링크 분석 또는 예시보기를 선택하시면 여기에 비디오 플레이어와 추천 질문 팁이 나타납니다.")


# ------------------------------------------------------------------
# 하단 오른쪽: 대화방 세션 관리 및 챗봇 창
# ------------------------------------------------------------------
with bot_col2:
    st.markdown(f"##### 🤖 {st.session_state.current_session_id}")
    
    # 세션 컨트롤 라인
    session_ctrl_col1, session_ctrl_col2 = st.columns([1, 1.5])
    
    with session_ctrl_col1:
        if st.button("➕ 새 대화 시작하기", use_container_width=True, type="primary"):
            if st.session_state.chat_history:
                st.session_state.chat_sessions[st.session_state.current_session_id] = st.session_state.chat_history
            
            new_id = f"대화방 #{len(st.session_state.chat_sessions) + 1}"
            st.session_state.current_session_id = new_id
            st.session_state.chat_history = [] 
            st.rerun()
            
    with session_ctrl_col2:
        if st.session_state.chat_sessions:
            options = ["선택 안 함"] + list(st.session_state.chat_sessions.keys())
            selected_room = st.selectbox(
                "📂 이전 대화 불러오기:",
                options=options,
                index=0,
                label_visibility="collapsed"
            )
            if selected_room != "선택 안 함" and selected_room != st.session_state.current_session_id:
                st.session_state.chat_sessions[st.session_state.current_session_id] = st.session_state.chat_history
                st.session_state.current_session_id = selected_room
                st.session_state.chat_history = st.session_state.chat_sessions[selected_room]
                st.rerun()

    st.write("") 
    
    # 대화록 렌더링
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    # 채팅 텍스트 인풋창
    if user_question := st.chat_input("분석된 내용에 대해 질문하세요!"):
        with st.chat_message("user"):
            st.markdown(user_question)
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        
        with st.chat_message("assistant"):
            if not st.session_state.processed_video:
                response = "⚠️ 먼저 상단 화면에서 분석을 완료해 주세요!"
                st.markdown(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
            else:
                with st.spinner("답변 생성 중..."):
                    response = run_rag_inference(user_question)
                    st.markdown(response)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})