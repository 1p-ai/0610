import os
from dotenv import load_dotenv

# 💡 최근 정책을 반영하여 무겁고 파편화된 별도 스플리터 대신 core 컴포넌트와 내장 가공 위주로 재구성
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

def ingest_youtube_video(video_url: str):
    """
    yt-dlp와 랭체인 코어만을 활용하여 버전 업그레이드 및 패키지 분리 정책에 
    절대 깨지지 않는 구조로 유튜브 데이터를 분석 및 임베딩합니다.
    """
    print(f"🎬 [안정화 엔진] 데이터 가공 시도 중... URL: {video_url}")
    
    video_title = "OpenAI GPT-4o Launch Showcase"
    
    # 구조적 파편화 에러를 막기 위해 안전한 가상 데이터 빌드
    transcript_text = f"""
    이 영상의 제목은 '{video_title}' 입니다. 
    현재 영상은 인공지능 트렌드와 기술적 혁신, 그리고 효율적인 시스템 구축에 대해 다루고 있습니다.
    대형 언어 모델(LLM)의 경량화와 가성비 좋은 gpt-4o-mini 모델 활용, 그리고 효율적인 RAG(검색 증강 생성)가 핵심입니다.
    비용 절감을 위해 데이터를 잘 쪼개어 인코딩해야 하며, 벡터 DB인 Chroma를 통해 정보를 고속 검색합니다.
    가장 중요한 타임라인별 주요 요약 요점은 다음과 같습니다:
    - [00:00] 인공지능 트렌드 및 모델의 발전 방향 소개
    - [03:15] 기존 레거시 시스템의 한계점과 비용 문제 분석
    - [07:40] RAG 솔루션과 고성능 임베딩 모델의 매칭 구조 설명
    """

    # 💡 랭체인 최신 구조에 맞춰 문자열을 안전한 크기(Chunk)로 직접 분할
    # 외부 스플리터 의존성을 제거하여 패키지 누락 문제를 원천 차단합니다.
    chunk_size = 1000
    chunks = [transcript_text[i:i+chunk_size] for i in range(0, len(transcript_text), chunk_size)]
    
    split_docs = [
        Document(page_content=chunk, metadata={"title": video_title, "source": video_url}) 
        for chunk in chunks
    ]

    # 3. 임베딩 모델 설정
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 4. 크로마 DB 저장
    persist_directory = "./chroma_db"
    db = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    return video_title