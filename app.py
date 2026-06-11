import streamlit as st

st.set_page_config(
    page_title="AI Agent Project",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 AI Agent 프로젝트 메인 페이지")
st.markdown("""
### 안녕하세요! 이 프로젝트에 포함된 다양한 AI 에이전트 앱을 탐색해보세요.

왼쪽 사이드바에서 원하는 앱을 선택하여 실행할 수 있습니다.

**사용 가능한 앱 목록:**
- **📄 PDF Reader:** PDF 파일을 업로드하고 내용에 대해 질문합니다.
- **🤖 RAG Chatbot:** 로컬 DB 또는 실시간으로 업로드한 문서를 기반으로 대화하는 고급 챗봇입니다.
- **🎬 YouTube Q&A:** 유튜브 영상 링크를 분석하고 요약 및 질의응답을 제공합니다.

**시작하려면 사이드바에서 앱을 선택하세요.**
""")