# ingest.py
import os
from dotenv import load_dotenv
import pypdf

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# 1. 최신 가이드라인 반영: pypdf로 PDF 텍스트 직접 추출
print("📄 PDF 문서 읽는 중...")
pdf_reader = pypdf.PdfReader("unsu.pdf")
text = ""
for page in pdf_reader.pages:
    text += page.extract_text() or ""
    
docs = [Document(page_content=text, metadata={"source": "unsu.pdf"})]

# 2. 지정하신 설정값으로 텍스트 분할
print("✂️ 텍스트 청크 분할 중...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 20,
    length_function = len,
    is_separator_regex = False
)
texts = text_splitter.split_documents(docs)

# 3. OpenAI 임베딩 및 Chroma 로컬 디스크 저장 (persist_directory 지정)
print("🧠 벡터 데이터베이스(Chroma) 빌드 및 저장 중...")
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma.from_documents(
    documents=texts, 
    embedding=embeddings_model,
    persist_directory="./chroma_db"  # 이 폴더 안에 데이터가 영구 보존됩니다.
)

print("✨ 성공적으로 chroma_db 폴더에 저장되었습니다!")