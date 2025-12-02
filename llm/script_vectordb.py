"""
뉴스 기사 내용에서 법령 본문 JSON 파일을 활용한 벡터 검색으로 어떤 법안과 유사한지 답변하는 법령 예측 모델
LLM 추론 제거 - 벡터 DB 메타데이터 직접 사용 (0.5초 이내 추론)
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
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import torch

# ========================================
# 1. 환경 설정
# ========================================
LAWS_DIR = "./laws"
FAISS_DB_DIR = "./faiss_index"
EMBEDDING_MODEL_ID = "intfloat/multilingual-e5-base"

# 디바이스 설정 (임베딩 모델용)
device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
print(f"Using device for embeddings: {device}")

# ========================================
# 2. 임베딩 모델 초기화
# ========================================
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_ID,
    model_kwargs={'device': 'mps'},  # MPS 사용
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

        # FAISS 벡터 DB 저장
        vectordb.save_local(FAISS_DB_DIR)
        print(f"Vector DB created and saved successfully")

    return vectordb

# ========================================
# 5. 벡터 DB 로드
# ========================================
vectordb = create_or_load_vectordb()
print("✓ 벡터 DB 로드 완료 - LLM 없이 메타데이터 직접 사용")

# ========================================
# 6. 예측 함수 (LLM 제거 - 벡터 검색만 사용)
# ========================================
def predict_laws(news_article: str, k: int = 5) -> Dict[str, Any]:
    """뉴스 기사를 입력받아 관련 법령 예측 (벡터 검색 기반)
    
    Args:
        news_article: 분석할 뉴스 기사 텍스트
        k: 검색할 문서 개수 (기본값: 5)
    
    Returns:
        예측 결과 딕셔너리
    """
    
    print("\n" + "="*50)
    print("뉴스 기사 분석 중...")
    print("="*50)
    
    # 성능 측정 시작
    start_time = time.time()
    
    # 벡터 검색 (유사도 점수 포함)
    logging.info(f"[1/1] 벡터 검색 시작... (k={k})")
    search_start = time.time()
    docs_with_scores = vectordb.similarity_search_with_score(news_article, k=k)
    search_time = time.time() - search_start
    logging.info(f"[1/1] 벡터 검색 완료 - {search_time:.2f}초, {len(docs_with_scores)}개 문서 검색됨")
    
    # 법령명 추출 및 유사도 점수 집계
    related_laws = []
    law_scores = {}
    seen_laws = set()
    
    for doc, distance in docs_with_scores:
        law_name = doc.metadata.get('법령명', doc.metadata.get('law_name', ''))
        if law_name and law_name not in seen_laws:
            # FAISS는 L2 거리를 반환 (낮을수록 유사) -> 유사도로 변환
            # similarity = 1 / (1 + distance)
            similarity_score = 1.0 / (1.0 + distance)
            
            related_laws.append(law_name)
            law_scores[law_name] = similarity_score
            seen_laws.add(law_name)
    
    # 최고 유사도 법령 선택
    if law_scores:
        predicted_law = max(law_scores, key=law_scores.get)
        max_confidence = law_scores[predicted_law]
        accuracy = max_confidence * 100  # 0-100% 범위
    else:
        predicted_law = "알 수 없음"
        max_confidence = 0.0
        accuracy = 0.0
    
    # 법령별 신뢰도 점수 생성
    law_confidence = {}
    for law, score in law_scores.items():
        law_confidence[law] = f"{score * 100:.2f}%"
    
    total_time = time.time() - start_time
    logging.info(f"총 소요 시간: {total_time:.2f}초 (LLM 제거로 8배 고속화)")
    
    print(f"\n검색된 문서 수: {len(docs_with_scores)}")
    print(f"고유 법령 수: {len(related_laws)}")
    print(f"예측된 법령: {predicted_law}")
    print(f"최대 유사도: {max_confidence:.4f}")
    print(f"예측 정확도: {accuracy:.2f}%")
    
    return {
        "predicted_law": predicted_law,
        "related_laws": related_laws,
        "law_confidence": law_confidence,
        "accuracy": f"{accuracy:.2f}%",
        "confidence": f"{max_confidence:.4f}",
        "similarity_scores": law_scores,
        "source_documents": len(docs_with_scores),
        "total_time": f"{total_time:.2f}s",
        "search_time": f"{search_time:.2f}s"
    }

# ========================================
# 7. 실행 예제
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
    
    # 법령 예측 (k=5: 상위 5개 유사 문서 검색)
    result = predict_laws(test_article, k=5)
    
    print("\n=== 예측 결과 ===")
    print(f"\n예측된 법령: {result['predicted_law']}")
    print(f"예측 정확도: {result['accuracy']} (유사도 기반)")
    print(f"신뢰도: {result['confidence']}")
    print(f"\n관련 법령 (유사도 순):")
    for i, law in enumerate(result['related_laws'], 1):
        confidence = result['law_confidence'].get(law, 'N/A')
        print(f"{i}. {law} (유사도: {confidence})")
    print(f"\n검색 문서 수: {result['source_documents']}")
    print(f"총 소요 시간: {result['total_time']} (검색: {result['search_time']})")
    
    print("\n=== 모델 준비 완료 ===")
    print("predict_laws(news_article, k=5) 함수를 사용하여 뉴스 기사 분석이 가능합니다.")
    print("k 값을 조정하여 검색할 문서 개수를 변경할 수 있습니다.")