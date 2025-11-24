import json
import os
import requests
from konlpy.tag import Okt
import asyncio
import time
import re
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import itertools
import multiprocessing as mp
import threading

import torch
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import matplotlib.pyplot as plt

# tokenizers 병렬 처리 충돌 방지
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ========================================
# Singleton Tokenizer with Thread Safety
# ========================================
class TokenizerSingleton:
    """스레드 세이프 싱글톤 토크나이저"""
    _instance = None
    _lock = threading.Lock()
    _tokenizer = None
    _model_ckpt = None
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(TokenizerSingleton, cls).__new__(cls)
        return cls._instance
    
    def get_tokenizer(self, model_ckpt='gogamza/kobart-base-v2'):
        """토크나이저를 반환 (필요시 초기화)"""
        if self._tokenizer is None or self._model_ckpt != model_ckpt:
            with self._lock:
                # Double-checked locking
                if self._tokenizer is None or self._model_ckpt != model_ckpt:
                    print(f"토크나이저 초기화 중: {model_ckpt}")
                    self._tokenizer = AutoTokenizer.from_pretrained(model_ckpt, force_download=True)
                    self._model_ckpt = model_ckpt
                    print("토크나이저 초기화 완료!")
        return self._tokenizer

# 전역 토크나이저 인스턴스
tokenizer_singleton = TokenizerSingleton()

def tokenize_summarize_dataset(dataset, model_ckpt='gogamza/kobart-base-v2'):
    tokenizer = tokenizer_singleton.get_tokenizer(model_ckpt)
    body_tokens = []
    summarize_tokens = []
    for data in dataset:
        body_tokens.append(tokenizer(data['body']))
        summarize_tokens.append(tokenizer(data['summarize']))
    return {'body': body_tokens, 'summarize': summarize_tokens}


def filter_text(lines):
    """
    기자/이메일 포함 문장 제거
    """
    reporter_pattern = r'[가-힣]{2,4}\s+기자'
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'

    def should_keep(segment):
        """기자/이메일이 없으면 True"""
        return (not re.search(reporter_pattern, segment) and 
                not re.search(email_pattern, segment))

    return [line for line in lines if should_keep(line)]

def cleanse_single_article_async(data_tuple):
    """단일 기사를 처리하는 함수 (비동기 처리용)"""
    article, idx = data_tuple
    
    print(f"{idx + 1}번 기사 본문 클린징 작업 중...")
    
    try:
        # 원본 데이터 복사하여 안전하게 처리
        article_copy = article.copy()
        filtered_body_sentences = filter_text(article_copy['body'])
        
        # 빈 리스트 체크
        if not filtered_body_sentences:
            article_copy['body'] = ""
            print(f"{idx + 1}번 기사 본문 클린징 완료 (빈 본문)")
            return idx, article_copy
        
        tokenizer = tokenizer_singleton.get_tokenizer('gogamza/kobart-base-v2')
        tokenized = tokenizer(filtered_body_sentences)
        
        # input_ids가 비어있는지 확인
        if not tokenized.get('input_ids'):
            article_copy['body'] = ""
            print(f"{idx + 1}번 기사 본문 클린징 완료 (토큰화 실패)")
            return idx, article_copy
        
        # 토큰 길이 계산
        total_tokens = len(list(itertools.chain(*tokenized['input_ids'])))
        article_copy['body'] = filtered_body_sentences if total_tokens <= 1024 else ""
        
        print(f"{idx + 1}번 기사 본문 클린징 작업 완료! (토큰 수: {total_tokens})")
        return idx, article_copy
        
    except (IndexError, KeyError, TypeError) as e:
        print(f"⚠️  {idx + 1}번 기사 처리 중 오류 발생: {type(e).__name__} - {str(e)}")
        article_copy = article.copy()
        article_copy['body'] = ""  # 오류 발생 시 빈 본문으로 설정
        return idx, article_copy
    except Exception as e:
        print(f"❌ {idx + 1}번 기사 처리 중 예상치 못한 오류 발생: {type(e).__name__} - {str(e)}")
        article_copy = article.copy()
        article_copy['body'] = ""  # 오류 발생 시 빈 본문으로 설정
        return idx, article_copy

async def process_batch_async(batch, semaphore):
    """세마포어를 사용해 동시 실행 수를 제한하면서 배치를 처리합니다."""
    async with semaphore:
        # asyncio.to_thread를 사용하여 스레드에서 실행 (JVM 안전)
        tasks = [asyncio.to_thread(cleanse_single_article_async, article_data) for article_data in batch]
        return await asyncio.gather(*tasks)
        

async def cleanse_articles_async_batch(dataset, batch_size=50, max_concurrent=5):
    """비동기 배치 처리로 모든 기사를 클린징합니다."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # (article, idx) 튜플 리스트 생성
    articles_with_idx = [(dataset[idx], idx) for idx in range(len(dataset))]
    
    # 데이터를 배치로 나누기
    batches = [articles_with_idx[i:i + batch_size] for i in range(0, len(articles_with_idx), batch_size)]
    
    print(f"총 {len(batches)}개의 배치로 나누어 처리합니다 (배치 크기: {batch_size}, 최대 동시 실행: {max_concurrent})")
    print(f"사용 가능한 CPU 코어 수: {mp.cpu_count()}개")
    
    # 배치별로 비동기 처리
    all_results = []
    for i, batch in enumerate(batches):
        print(f"배치 {i+1}/{len(batches)} 처리 중...")
        batch_results = await process_batch_async(batch, semaphore)
        all_results.extend(batch_results)
    
    # 결과를 원래 순서대로 정렬
    all_results.sort(key=lambda x: x[0])  # idx로 정렬
    
    return [result[1] for result in all_results]  # article만 반환

def cleanse_single_article_thread(data_tuple):
    """단일 기사를 처리하는 함수 (프로세스/스레드 병렬 처리용)"""
    import requests
    from konlpy.tag import Okt
    
    article, idx = data_tuple
    
    # 각 프로세스/스레드에서 독립적으로 불용어와 okt 초기화
    try:
        url = "https://raw.githubusercontent.com/stopwords-iso/stopwords-ko/master/stopwords-ko.txt"
        korean_stopwords = set(requests.get(url).text.split("\n"))
        for stopword in ["은", "는", "이", "가", "고", "정말", "의", "이나", "이라고", "인", "이다", "하여", "``", "에", "에는"]:
            korean_stopwords.add(stopword) 
        korean_stopwords.remove("모")
    except:
        # 네트워크 오류 시 기본 불용어 사용
        korean_stopwords = {"은", "는", "이", "가", "고", "정말", "의", "이나", "이라고", "인", "이다", "하여", "``", "에", "에는"}
    
    okt = Okt()
    
    def clean_text_local(text):
        if not isinstance(text, str):
            return ""

        """텍스트 전처리 함수"""
        splited_texts = text.split()
        okt_text = okt.morphs(text)
        cleaned_texts = []
        temp = ""
        pointer = 0
        for split_text in splited_texts:
            while pointer < len(okt_text) and okt_text[pointer] in split_text:
                # 불용어 제거
                if not okt_text[pointer] in korean_stopwords:
                    temp += okt_text[pointer]
                pointer += 1
            if temp: cleaned_texts.append(temp)
            temp = ""
        return " ".join(cleaned_texts)
    
    print(f"{idx + 1}번 기사 본문 클린징 작업 중...")
    
    try:
        # 원본 데이터 복사하여 안전하게 처리
        article_copy = article.copy()
        filtered_body_sentences = filter_text(article_copy['body'])
        
        # 빈 리스트 체크
        if not filtered_body_sentences:
            article_copy['body'] = ""
            print(f"{idx + 1}번 기사 본문 클린징 완료 (빈 본문)")
            return idx, article_copy
        
        tokenizer = tokenizer_singleton.get_tokenizer('gogamza/kobart-base-v2')
        tokenized = tokenizer(filtered_body_sentences)
        
        # input_ids가 비어있는지 확인
        if not tokenized.get('input_ids'):
            article_copy['body'] = ""
            print(f"{idx + 1}번 기사 본문 클린징 완료 (토큰화 실패)")
            return idx, article_copy
        
        # 토큰 길이 계산
        total_tokens = len(list(itertools.chain(*tokenized['input_ids'])))
        article_copy['body'] = filtered_body_sentences if total_tokens <= 1024 else ""
        
        print(f"{idx + 1}번 기사 본문 클린징 작업 완료! (토큰 수: {total_tokens})")
        return idx, article_copy
        
    except (IndexError, KeyError, TypeError) as e:
        print(f"⚠️  {idx + 1}번 기사 처리 중 오류 발생: {type(e).__name__} - {str(e)}")
        article_copy = article.copy()
        article_copy['body'] = ""  # 오류 발생 시 빈 본문으로 설정
        return idx, article_copy
    except Exception as e:
        print(f"❌ {idx + 1}번 기사 처리 중 예상치 못한 오류 발생: {type(e).__name__} - {str(e)}")
        article_copy = article.copy()
        article_copy['body'] = ""  # 오류 발생 시 빈 본문으로 설정
        return idx, article_copy

async def process_batch_async(batch, semaphore):
    """세마포어를 사용해 동시 실행 수를 제한하면서 배치를 처리합니다."""
    async with semaphore:
        # ThreadPoolExecutor를 사용해 CPU 집약적 작업을 별도 스레드에서 실행
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
            tasks = [
                loop.run_in_executor(executor, cleanse_single_article_async, article_data)
                for article_data in batch
            ]
            return await asyncio.gather(*tasks)

async def cleanse_articles_async_batch(dataset, batch_size=50, max_concurrent=5):
    """비동기 배치 처리로 모든 기사를 클린징합니다."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # (article, idx) 튜플 리스트 생성
    articles_with_idx = [(dataset[idx], idx) for idx in range(len(dataset))]
    
    # 데이터를 배치로 나누기
    batches = [articles_with_idx[i:i + batch_size] for i in range(0, len(articles_with_idx), batch_size)]
    
    print(f"총 {len(batches)}개의 배치로 나누어 처리합니다 (배치 크기: {batch_size}, 최대 동시 실행: {max_concurrent})")
    print(f"사용 가능한 CPU 코어 수: {mp.cpu_count()}개")
    
    # 배치별로 비동기 처리
    all_results = []
    for i, batch in enumerate(batches):
        print(f"배치 {i+1}/{len(batches)} 처리 중...")
        batch_results = await process_batch_async(batch, semaphore)
        all_results.extend(batch_results)
    
    # 결과를 원래 순서대로 정렬
    all_results.sort(key=lambda x: x[0])  # idx로 정렬
    
    return [result[1] for result in all_results]  # article만 반환

def cleanse_articles_parallel_process(dataset):
    """프로세스 병렬 처리로 모든 기사를 클린징 (로컬 환경용)"""
    # CPU 집약적 작업(형태소 분석, 텍스트 처리)이므로 ProcessPoolExecutor 사용
    max_workers = min(mp.cpu_count(), len(dataset))  # CPU 코어 수만큼

    print(f"프로세스 풀 크기: {max_workers}개")
    print(f"사용 가능한 CPU 코어 수: {mp.cpu_count()}개")

    # (article, idx) 튜플 리스트 생성
    articles_with_idx = [(dataset[idx], idx) for idx in range(len(dataset))]

    # ProcessPoolExecutor 사용하여 병렬 처리
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 모든 기사를 병렬로 처리
        results = list(executor.map(cleanse_single_article_thread, articles_with_idx))

    # 결과를 원래 순서대로 정렬
    results.sort(key=lambda x: x[0])  # idx로 정렬

    return [result[1] for result in results]  # article만 반환

async def cleanse_articles_parallel_thread(dataset):
    """스레드 병렬 처리로 모든 기사를 클린징 (노트북 환경용)"""
    # I/O 집약적 작업(requests, file I/O)이 많으므로 ThreadPoolExecutor 사용
    max_workers = min(mp.cpu_count() * 2, len(dataset), 8)  # CPU 코어의 2배까지, 최대 8개

    print(f"스레드 풀 크기: {max_workers}개")

    # (article, idx) 튜플 리스트 생성
    articles_with_idx = [(dataset[idx], idx) for idx in range(len(dataset))]

    # asyncio의 run_in_executor를 사용해 스레드 병렬 처리
    loop = asyncio.get_event_loop()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 모든 기사를 병렬로 처리
        tasks = [
            loop.run_in_executor(executor, cleanse_single_article_thread, article_data)
            for article_data in articles_with_idx
        ]

        # 모든 작업 완료까지 대기
        results = await asyncio.gather(*tasks)

    # 결과를 원래 순서대로 정렬
    results.sort(key=lambda x: x[0])  # idx로 정렬

    return [result[1] for result in results]  # article만 반환

# 환경에 따라 선택적으로 사용
async def cleanse_articles_parallel(dataset, model_ckpt=None, use_async_batch=True):
    """환경에 맞는 병렬 처리 선택"""
    if use_async_batch:
        # 비동기 배치 처리 사용 (가장 빠름)
        return await cleanse_articles_async_batch(dataset, batch_size=50, max_concurrent=mp.cpu_count())
    else:
        # 프로세스 병렬 처리 사용
        return cleanse_articles_parallel_process(dataset)

async def main_async():
    """비동기 메인 함수"""
    # ========================================
    # 1. 설정 및 초기화
    # ========================================
    # print("Okt 및 불용어 초기화 중...")
    # JVM 충돌 방지를 위해 먼저 전역 Okt 인스턴스 초기화
    # get_okt_and_stopwords()
    # print("Okt 초기화 완료!")
    
    model_ckpt = 'gogamza/kobart-base-v2'

    # ========================================
    # 2. 데이터 로드 및 전처리
    # ========================================

    with open('./newspaper_summarize_jsonl/newspaper_summarize.jsonl') as f:
        dataset = [json.loads(line) for line in f]
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42, shuffle=True)
    valid_dataset, test_dataset = train_test_split(test_dataset, test_size=0.5, random_state=42, shuffle=True)

    # 병렬 처리 시작 시간 기록
    start_time = time.time()

    print("비동기 배치 처리로 훈련 데이터셋 클린징 시작...")
    print(f"사용 가능한 CPU 코어 수: {mp.cpu_count()}개")
    print(f"처리할 기사 수: {len(train_dataset)}개")

    # 2. 검증 데이터셋 처리
    print("\n=== 검증 데이터셋 클린징 시작 ===")
    valid_start_time = time.time()
    cleaned_valid_dataset = await cleanse_articles_parallel(valid_dataset, model_ckpt, use_async_batch=True)
    cleaned_valid_dataset = [dataset for dataset in cleaned_valid_dataset if len(dataset['body']) > 0]
    valid_end_time = time.time()
    valid_processing_time = valid_end_time - valid_start_time
    print(f"검증 데이터셋 클린징 완료! 소요 시간: {valid_processing_time:.2f}초")
    print(f"처리된 기사 수: {len(cleaned_valid_dataset)}개")
    print(f"평균 처리 속도: {len(cleaned_valid_dataset)/valid_processing_time:.2f}개/초")

    # 저장 디렉토리 생성
    os.makedirs('./cleaned_datasets_v2', exist_ok=True)

    with open('./cleaned_datasets_v2/cleaned_valid_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(cleaned_valid_dataset, f, ensure_ascii=False, indent=2)
    print(f"✅ 클린징된 검증 데이터셋 저장 완료: {len(cleaned_valid_dataset)}개 기사")

    # 1. 훈련 데이터셋 처리
    print("\n=== 훈련 데이터셋 클린징 시작 ===")
    cleaned_train_dataset = await cleanse_articles_parallel(train_dataset, model_ckpt, use_async_batch=True)
    cleaned_train_dataset = [dataset for dataset in cleaned_train_dataset if len(dataset['body']) > 0]
    
    # 처리 시간 출력
    end_time = time.time()
    processing_time = end_time - start_time
    print(f"훈련 데이터셋 클린징 완료! 소요 시간: {processing_time:.2f}초")
    print(f"처리된 기사 수: {len(cleaned_train_dataset)}개")
    print(f"평균 처리 속도: {len(cleaned_train_dataset)/processing_time:.2f}개/초")
    
    # 클린징된 데이터셋 저장
    with open('./cleaned_datasets_v2/cleaned_train_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(cleaned_train_dataset, f, ensure_ascii=False, indent=2)
    print(f"✅ 클린징된 훈련 데이터셋 저장 완료: {len(cleaned_train_dataset)}개 기사")

    # 3. 테스트 데이터셋 처리
    print("\n=== 테스트 데이터셋 클린징 시작 ===")
    test_start_time = time.time()
    cleaned_test_dataset = await cleanse_articles_parallel(test_dataset, model_ckpt, use_async_batch=True)
    cleaned_test_dataset = [dataset for dataset in cleaned_test_dataset if len(dataset['body']) > 0]
    test_end_time = time.time()
    test_processing_time = test_end_time - test_start_time
    print(f"테스트 데이터셋 클린징 완료! 소요 시간: {test_processing_time:.2f}초")
    print(f"처리된 기사 수: {len(cleaned_test_dataset)}개")
    print(f"평균 처리 속도: {len(cleaned_test_dataset)/test_processing_time:.2f}개/초")

    with open('./cleaned_datasets_v2/cleaned_test_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(cleaned_test_dataset, f, ensure_ascii=False, indent=2)
    print(f"✅ 클린징된 테스트 데이터셋 저장 완료: {len(cleaned_test_dataset)}개 기사")

    # 전체 처리 시간
    total_processing_time = processing_time + valid_processing_time + test_processing_time
    total_articles = len(cleaned_train_dataset) + len(cleaned_valid_dataset) + len(cleaned_test_dataset)
    print(f"\n=== 전체 처리 완료 ===")
    print(f"전체 소요 시간: {total_processing_time:.2f}초")
    print(f"전체 처리된 기사 수: {total_articles}개")
    print(f"전체 평균 처리 속도: {total_articles/total_processing_time:.2f}개/초")

    # 데이터셋 크기 정보 저장
    dataset_info = {
        "train_size": len(cleaned_train_dataset),
        "valid_size": len(cleaned_valid_dataset),
        "test_size": len(cleaned_test_dataset),
        "total_size": total_articles,
        "processing_time": {
            "train_time": processing_time,
            "valid_time": valid_processing_time,
            "test_time": test_processing_time,
            "total_time": total_processing_time
        },
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open('./cleaned_datasets_v2/dataset_info.json', 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    print(f"✅ 데이터셋 정보 저장 완료")
    
    print(f"\n🎉 모든 처리 및 저장 완료!")
    print(f"📁 저장 위치: ./cleaned_datasets_v2/")
    print(f"📊 총 처리 시간: {total_processing_time:.2f}초")
    
    return cleaned_train_dataset, cleaned_valid_dataset, cleaned_test_dataset


if __name__ == "__main__":
    # 비동기 메인 함수 실행
    print("=== 비동기 배치 처리 모드로 실행 ===")
    asyncio.run(main_async())
