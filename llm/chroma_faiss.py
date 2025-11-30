from langchain_community.vectorstores import Chroma, FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

EMBEDDING_MODEL_ID = "intfloat/multilingual-e5-base"
# 기존 Chroma 로드
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_ID,
    model_kwargs={'device': 'cpu'},  # MPS 안정성 이슈로 CPU 사용
    encode_kwargs={'normalize_embeddings': True}
)

chroma_db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# 모든 문서 가져오기
documents = chroma_db.get()


docs = [
    Document(page_content=text, metadata=metadata)
    for text, metadata in zip(documents['documents'], documents['metadatas'])
]

faiss_db = FAISS.from_documents(docs, embeddings)

# 로컬에 저장
faiss_db.save_local("./faiss_index")