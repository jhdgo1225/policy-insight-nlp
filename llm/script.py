"""
뉴스 기사 내용에서 법령 본문 JSON 파일을 활용한 RAG로 어떤 법안과 유사한지 답변하는 법령 예측 모델
"""

import os
import json
import glob
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
import time
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# LangChain 관련 임포트
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.cache import InMemoryCache
from langchain_core.globals import set_llm_cache
from langchain_core.runnables import RunnableConfig

# Transformers 임포트
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, pipeline
import torch

# ========================================
# 1. 환경 설정
# ========================================
LAWS_DIR = "./llm/laws"
CHROMA_DB_DIR = "./llm/chroma_db"
FAISS_DB_DIR = "./llm/faiss_index"
LLM_MODEL_ID = "./llm/Meta-Llama-3-8B-Instruct"
# LAWS_DIR = "./laws"
# CHROMA_DB_DIR = "./chroma_db"
# FAISS_DB_DIR = "./faiss_index"
# LLM_MODEL_ID = "./Meta-Llama-3-8B-Instruct"
EMBEDDING_MODEL_ID = "intfloat/multilingual-e5-base"

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ========================================
# 2. 임베딩 모델 초기화
# ========================================
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_ID,
    model_kwargs={'device': 'cpu'},  # MPS 안정성 이슈로 CPU 사용
    encode_kwargs={'normalize_embeddings': True}
)

# ========================================
# 3. 법령 JSON 파일 로드 및 전처리
# ========================================
def load_law_documents() -> List[Document]:
    """법령 JSON 파일들을 로드하고 Document 객체로 변환"""
    documents = []
    json_files = glob.glob(os.path.join(LAWS_DIR, "*.json"))
    
    print(f"Found {len(json_files)} law files")
    
    for idx, json_file in enumerate(json_files):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                law_data = json.load(f)
                
            # 법령명 추출 (파일명에서)
            law_name = Path(json_file).stem
            
            # 법령 내용 추출 및 결합
            content_parts = []
            
            # JSON 구조 확인: 최상위에 "법령" 키가 있는지 확인
            if '법령' in law_data:
                law_info = law_data['법령']
                
                # 법령명 추출 (기본정보 > 법령명_한글)
                if '기본정보' in law_info:
                    basic_info = law_info['기본정보']
                    law_name_korean = basic_info.get('법령명_한글', '') or basic_info.get('법령명_한자', '')
                    if law_name_korean:
                        content_parts.append(f"법령명: {law_name_korean}")
                
                # 조문들 처리 (조문 > 조문단위)
                if '조문' in law_info and '조문단위' in law_info['조문']:
                    articles = law_info['조문']['조문단위']
                    if isinstance(articles, list):
                        for article in articles:
                            if isinstance(article, dict):
                                조문번호 = article.get('조문번호', '')
                                조문제목 = article.get('조문제목', '')
                                조문내용 = article.get('조문내용', '')
                                
                                # 항 정보 추가
                                if '항' in article:
                                    if isinstance(article['항'], list):
                                        for 항 in article['항']:
                                            if isinstance(항, dict):
                                                조문내용 += f"\n{항.get('항내용', '')}"
                                
                                article_text = f"\n제{조문번호}조 {조문제목}\n{조문내용}"
                                content_parts.append(article_text)
                
                # 제개정이유 추가 (선택적)
                if '제개정이유' in law_info and '제개정이유내용' in law_info['제개정이유']:
                    reason_content = law_info['제개정이유']['제개정이유내용']
                    if isinstance(reason_content, list) and len(reason_content) > 0:
                        if isinstance(reason_content[0], list):
                            reason_text = "\n".join(reason_content[0])
                            content_parts.append(f"\n제개정이유:\n{reason_text}")
            
            # 전체 내용 결합
            full_content = "\n".join(content_parts)
            
            # 빈 문서 검증 강화
            if full_content.strip() and len(full_content.strip()) > 10:
                # 메타데이터 추출
                법령명 = law_name
                if '법령' in law_data and '기본정보' in law_data['법령']:
                    법령명 = law_data['법령']['기본정보'].get('법령명_한글', law_name)
                
                doc = Document(
                    page_content=full_content,
                    metadata={
                        "law_name": law_name,
                        "source": json_file,
                        "법령명": 법령명
                    }
                )
                documents.append(doc)
                
                # 진행 상황 표시 (100개마다)
                if (idx + 1) % 100 == 0:
                    print(f"Processed {idx + 1}/{len(json_files)} files...")
            else:
                print(f"Skipping empty document: {law_name} (content length: {len(full_content.strip())})")
                
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"Loaded {len(documents)} law documents")
    
    if len(documents) == 0:
        raise ValueError("No valid documents loaded! Check JSON file structure and LAWS_DIR path.")
    
    return documents

# ========================================
# 4. 벡터 DB 생성 또는 로드
# ========================================
def create_or_load_vectordb() -> FAISS:
    """Faiss 벡터 DB 생성 또는 기존 DB 로드"""

    if os.path.exists(FAISS_DB_DIR) and os.listdir(FAISS_DB_DIR):
        print("Loading existing Faiss DB...")

        # FAISS 로드
        vectordb = FAISS.load_local(
            FAISS_DB_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"Loaded {len(vectordb.index_to_docstore_id.keys())} documents from existing DB")
    else:
        print("Creating new Faiss DB...")
        
        # 법령 문서 로드
        documents = load_law_documents()
        
        # 텍스트 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

        print("Splitting documents into chunks...")
        splits = text_splitter.split_documents(documents)
        print(f"Created {len(splits)} text chunks")

        # 빈 청크 필터링
        valid_splits = [doc for doc in splits if doc.page_content.strip() and len(doc.page_content.strip()) > 10]
        print(f"Valid chunks after filtering: {len(valid_splits)}")

        if len(valid_splits) == 0:
            raise ValueError("No valid text chunks created! Check document content.")

        # 벡터 DB 생성 (배치 처리로 안정성 향상)
        print("Creating vector database (this may take a while)...")

        # 배치 크기 설정 (메모리 효율성)
        batch_size = 100
        total_batches = (len(valid_splits) + batch_size - 1) // batch_size

        # 첫 번째 배치로 DB 초기화
        first_batch = valid_splits[:batch_size]
        # FAISS 로드
        vectordb = FAISS.from_documents(
            documents=first_batch,
            embedding=embeddings,
            persist_directory=FAISS_DB_DIR
        )
        print(f"Initialized DB with first batch ({len(first_batch)} chunks)")

        # 나머지 배치 추가
        for i in range(1, total_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(valid_splits))
            batch = valid_splits[start_idx:end_idx]

            if batch:
                vectordb.add_documents(batch)
                print(f"Added batch {i+1}/{total_batches} ({len(batch)} chunks)")

        print(f"Vector DB created with {vectordb._collection.count()} chunks")

    return vectordb

# ========================================
# 5. LLM 모델 초기화
# ========================================
print("Loading LLM model...")

# 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
    low_cpu_mem_usage=True
)

# CPU로 강제 이동 (MPS 안정성)
if not torch.cuda.is_available():
    model = model.to('cpu')

# HuggingFacePipeline 초기화 (model_type 오류 해결)
# pipeline의 model.config가 제대로 설정되었는지 확인
print(f"Model config type: {type(model.config)}")
print(f"Model type from config: {model.config.model_type}")

# 토크나이저 로드 (로컬 경로 사용 시 token 파라미터 불필요)
tokenizer = AutoTokenizer.from_pretrained(
    LLM_MODEL_ID,
    trust_remote_code=True
)

# pad_token 설정 (데드락 및 경고 방지)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id


# Pipeline 생성
text_generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.1,
    do_sample=True,
    top_p=0.95,
    pad_token_id=tokenizer.eos_token_id  # pad_token_id 명시적 설정
)

# HuggingFacePipeline 초기화
# pipeline 파라미터만 전달 (model_id는 선택사항)
llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# 캐시 설정
set_llm_cache(InMemoryCache())
print("LLM cache enabled")

# ========================================
# 6. 벡터 DB 로드
# ========================================
vectordb = create_or_load_vectordb()

# ========================================
# 7. RAG 체인 구성
# ========================================
# 프롬프트 템플릿 설정 (최신 ChatPromptTemplate 사용)
system_prompt = """당신은 법령 전문가입니다. 주어진 뉴스 기사와 관련된 법령을 분석하고 예측해주세요.

다음 법령 정보를 참고하세요:
{context}

위 뉴스 기사와 가장 관련성이 높은 법령 3개를 순서대로 제시하고, 각 법령명과 관련 이유를 설명해주세요.

답변 형식:
1. [법령명]: 관련 이유
2. [법령명]: 관련 이유
3. [법령명]: 관련 이유"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# Retriever 설정 (유사도 점수 포함)
retriever = vectordb.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5} # 상위 5개 문서 검색
)

# Document chain 생성
document_chain = create_stuff_documents_chain(llm, prompt)

# Retrieval chain 생성 (최신 create_retrieval_chain 사용)
qa_chain = create_retrieval_chain(retriever, document_chain)

# ========================================
# 8. 예측 함수
# ========================================
def predict_laws(news_article: str) -> Dict[str, Any]:
    """뉴스 기사를 입력받아 관련 법령 예측"""
    
    print("\n" + "="*50)
    print("뉴스 기사 분석 중...")
    print("="*50)
    
    # 성능 측정 시작
    start_time = time.time()
    
    # 1단계: 벡터 검색 시간 측정
    logging.info("[1/3] 벡터 검색 시작...")
    search_start = time.time()
    retrieved_docs = retriever.invoke(news_article)
    search_time = time.time() - search_start
    logging.info(f"[1/3] 벡터 검색 완료 - {search_time:.2f}초, {len(retrieved_docs)}개 문서 검색됨")
    
    # 2단계: 프롬프트 생성 시간 측정
    logging.info("[2/3] 프롬프트 생성 중...")
    prompt_start = time.time()
    
    # *** 데드락 해결: qa_chain.invoke()만 한 번 호출 ***
    # vectordb에 중복 접근하지 않도록 수정
    logging.info("[3/3] LLM 추론 시작 (시간이 오래 걸릴 수 있습니다)...")
    llm_start = time.time()
    result = qa_chain.invoke({"input": news_article}, config=RunnableConfig(max_concurrency=1))
    llm_time = time.time() - llm_start
    logging.info(f"[3/3] LLM 추론 완료 - {llm_time:.2f}초")
    
    total_time = time.time() - start_time
    logging.info(f"총 소요 시간: {total_time:.2f}초 (검색: {search_time:.2f}초, LLM: {llm_time:.2f}초)")
    
    # 관련 문서 추출 (최신 API: context 키 사용)
    source_docs = result.get('context', [])
    
    # 법령명 추출 및 유사도 점수 매핑 (중복 제거)
    related_laws = []
    law_scores = {}
    seen_laws = set()
    
    # retriever가 이미 검색한 문서들에서 법령 추출
    # 별도의 similarity_search_with_score 호출 제거 (데드락 방지)
    for doc in source_docs:
        law_name = doc.metadata.get('법령명', doc.metadata.get('law_name', ''))
        if law_name and law_name not in seen_laws:
            related_laws.append(law_name)
            seen_laws.add(law_name)
            # 기본 신뢰도 점수 부여 (순서 기반)
            law_scores[law_name] = 1.0 - (len(related_laws) - 1) * 0.1
            
        if len(related_laws) >= 3:
            break
    
    # 정확도 계산 (검색된 문서 수 기반)
    if source_docs:
        # 검색 품질: 검색된 문서 수 / 요청한 문서 수
        retrieval_quality = len(source_docs) / 5.0
        
        # 신뢰도: 검색 품질 기반 (최소 50%, 최대 95%)
        confidence = 0.5 + (retrieval_quality * 0.45)
        
        # 정확도: 0-100% 범위로 변환
        accuracy = min(confidence * 100, 100)
        
        print(f"\n검색된 문서 수: {len(source_docs)}")
        print(f"검색 품질: {retrieval_quality:.2f}")
        print(f"신뢰도: {confidence:.4f}")
    else:
        accuracy = 0
        confidence = 0
    
    # 법령별 신뢰도 점수 생성
    law_confidence = {}
    for law in related_laws:
        if law in law_scores:
            law_confidence[law] = f"{law_scores[law] * 100:.2f}%"
    
    return {
        "answer": result['answer'],
        "related_laws": related_laws,
        "law_confidence": law_confidence,
        "accuracy": f"{accuracy:.2f}%",
        "confidence": f"{confidence:.4f}",
        "source_documents": len(source_docs)
    }

# ========================================
# 9. 실행 예제
# ========================================
if __name__ == "__main__":
    # 테스트 뉴스 기사
    test_article = """
    정부가 중소기업의 기술혁신을 촉진하기 위한 새로운 지원책을 발표했다. 
    이번 정책은 중소기업이 신기술 개발에 투자할 경우 세제 혜택을 제공하고, 
    연구개발비에 대한 보조금을 확대하는 내용을 담고 있다. 
    특히 벤처기업과 스타트업에 대한 지원이 강화될 예정이다.
    """
    
    print("\n=== 뉴스 기사 ===")
    print(test_article)
    
    # 법령 예측
    result = predict_laws(test_article)
    
    print("\n=== 예측 결과 ===")
    print(f"\n답변:\n{result['answer']}")
    print(f"\n관련 법령 (상위 3개):")
    for i, law in enumerate(result['related_laws'], 1):
        confidence = result['law_confidence'].get(law, 'N/A')
        print(f"{i}. {law} (신뢰도: {confidence})")
    print(f"\n검색된 문서 수: {result['source_documents']}")
    print(f"신뢰도: {result['confidence']}")
    print(f"답변 정확도: {result['accuracy']}")
    
    print("\n=== 모델 준비 완료 ===")
    print("predict_laws(news_article) 함수를 사용하여 뉴스 기사 분석이 가능합니다.")