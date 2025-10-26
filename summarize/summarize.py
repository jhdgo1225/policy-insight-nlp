import json
import os
import requests
from konlpy.tag import Okt
import time
import re
import copy

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from transformers import EarlyStoppingCallback
from datasets import Dataset
import evaluate
import numpy as np

# tokenizers 병렬 처리 충돌 방지
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 전역 변수로 Okt와 불용어를 한 번만 초기화 (JVM 충돌 방지)
_okt_instance = None
_korean_stopwords = None

def get_okt_and_stopwords():
    """전역 Okt 인스턴스와 불용어를 반환 (JVM 충돌 방지)"""
    global _okt_instance, _korean_stopwords
    
    if _okt_instance is None:
        from konlpy.tag import Okt
        _okt_instance = Okt()
        
    if _korean_stopwords is None:
        try:
            import requests
            url = "https://raw.githubusercontent.com/stopwords-iso/stopwords-ko/master/stopwords-ko.txt"
            _korean_stopwords = set(requests.get(url).text.split("\n"))
            for stopword in ["은", "는", "이", "가", "고", "정말", "의", "이나", "이라고", "인", "이다", "하여", "``", "에", "에는"]:
                _korean_stopwords.add(stopword) 
            _korean_stopwords.remove("모")
        except:
            # 네트워크 오류 시 기본 불용어 사용
            _korean_stopwords = {"은", "는", "이", "가", "고", "정말", "의", "이나", "이라고", "인", "이다", "하여", "``", "에", "에는"}
    
    return _okt_instance, _korean_stopwords


def tokenize_summarize_dataset(dataset, model_ckpt='gogamza/kobart-base-v2'):
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt, force_download=True)
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


def preprocess_text_for_inference(text_lines):
    """
    추론을 위한 텍스트 전처리 파이프라인
    1. 기자/이메일 제거
    2. 형태소 분석 및 불용어 제거
    """
    # 1. 기자/이메일 제거
    filtered_lines = filter_text(text_lines)
    
    # 2. 형태소 분석 및 불용어 제거
    okt, stopwords = get_okt_and_stopwords()
    
    processed_lines = []
    for line in filtered_lines:
        # 형태소 분석
        morphs = okt.morphs(line)
        # 불용어 제거
        filtered_morphs = [word for word in morphs if word not in stopwords]
        processed_line = ' '.join(filtered_morphs)
        processed_lines.append(processed_line)
    
    # 요약 태스크를 인식하기 위해 전처리 문장 리스트의 첫 번째 요소로 "요약: " 추가
    processed_lines.insert(0, "요약: ")
    return processed_lines

def func(x):
    result = copy.deepcopy(x)
    result['body'].insert(0, "요약: ")
    return result

def load_cleaned_datasets():
    """전처리된 데이터셋 로드"""
    with open('./cleaned_datasets/cleaned_train_dataset.json') as f:
        train_dataset = [data for data in json.load(f)]
        train_dataset = list(map(func, train_dataset))
    with open('./cleaned_datasets/cleaned_valid_dataset.json') as f:
        valid_dataset = [data for data in json.load(f)]
        valid_dataset = list(map(func, valid_dataset))
    with open('./cleaned_datasets/cleaned_test_dataset.json') as f:
        test_dataset = [data for data in json.load(f)]
        test_dataset = list(map(func, test_dataset))
    return train_dataset, valid_dataset, test_dataset


def train_model():
    """모델 훈련 함수"""
    print("="*50)
    print("KoBART 문서 요약 모델 훈련 시작")
    print("="*50)
    
    """
    1. 데이터셋 로드
    훈련, 검증, 테스트 데이터셋 구성
    - body: 전처리된 본문의 문장들의 집합. (예시 -> body: [[문장1], [문장2], [문장3], ... ])
    - summarize: 요약문 (예시 -> summarize: "요약문")
    """
    print("\n[1단계] 데이터셋 로드 중...")
    train_dataset, valid_dataset, test_dataset = load_cleaned_datasets()
    print(f"  - 훈련 데이터: {len(train_dataset)}개")
    print(f"  - 검증 데이터: {len(valid_dataset)}개")
    print(f"  - 테스트 데이터: {len(test_dataset)}개")

    """
    2. KoBART 모델(BART 계열) 훈련을 위한 입력, 레이블 데이터 토크나이징
    """
    print("\n[2단계] 데이터 토크나이징 중...")
    # 이 부분은 Trainer 내부에서 처리됨

    """
    3. KoBART 모델 로드 후 모델 초기화 및 하이퍼파라미터 설정
    """
    print("\n[3단계] KoBART 모델 로드 및 설정 중...")
    model_ckpt = 'gogamza/kobart-base-v2'
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt, force_download=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt)
    
    # GPU 사용 가능 여부 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"  - 사용 디바이스: {device}")
    print(f"  - 모델: {model_ckpt}")
    
    # 한국어 문서 요약 선례를 참고한 최적의 하이퍼파라미터 설정
    print("\n  [하이퍼파라미터 설정]")
    print("  - Learning rate: 3e-5 (한국어 문서 요약 연구에서 가장 일반적)")
    print("  - Batch size: 8 (GPU 메모리 고려)")
    print("  - Max epochs: 50 (Early Stopping으로 자동 조기 종료)")
    print("  - Early Stopping patience: 3 (검증 손실 개선 없으면 3 에폭 후 종료)")
    print("  - Max input length: 1024")
    print("  - Max summary length: 256")
    print("  - Gradient accumulation: 2")
    print("  - Warmup steps: 500")
    print("  ✨ Early Stopping 활성화: 과적합 방지 및 최적 모델 자동 선택")
    
    training_args = Seq2SeqTrainingArguments(
        output_dir='./kobart_summarization_results',
        num_train_epochs=50,                   # Early Stopping이 실제 종료 시점 결정
        per_device_train_batch_size=8,         # GPU 메모리에 따라 조정 (8-16)
        per_device_eval_batch_size=8,
        learning_rate=3e-5,                    # KoBART 문서 요약에서 가장 일반적인 값
        weight_decay=0.01,                     # 가중치 감쇠
        warmup_steps=500,                      # 학습률 워밍업
        logging_dir='./logs',
        logging_steps=100,
        save_strategy="epoch",                 # 에폭마다 저장 (eval_strategy와 일치)
        save_total_limit=3,                    # 최대 체크포인트 개수 (최근 3개만 유지)
        eval_strategy="epoch",                 # 에폭마다 평가
        load_best_model_at_end=True,           # 최적 모델 자동 로드
        metric_for_best_model="eval_loss",     # 검증 손실 기준
        greater_is_better=False,               # 손실은 낮을수록 좋음
        predict_with_generate=True,            # 생성 기반 평가
        generation_max_length=256,             # 요약문 최대 길이
        gradient_accumulation_steps=2,         # 배치 크기를 효과적으로 늘림
        fp16=torch.cuda.is_available(),        # Mixed precision (GPU에서만)
        report_to="none",                      # wandb 등 로깅 비활성화
    )
    
    # 데이터 전처리: body를 입력으로, summarize를 레이블로
    def preprocess_function(examples):
        # body는 문장 리스트이므로 join
        inputs = [' '.join(body) if isinstance(body, list) else body for body in examples['body']]
        targets = examples['summarize']
        
        # 토크나이징
        model_inputs = tokenizer(
            inputs, 
            max_length=1024,          # 입력 최대 길이
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # 레이블 토크나이징
        labels = tokenizer(
            targets,
            max_length=256,           # 요약문 최대 길이
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    
    # 데이터셋을 Hugging Face Dataset 형식으로 변환
    print("\n  [데이터셋 변환 중...]")
    train_hf = Dataset.from_dict({
        'body': [d['body'] for d in train_dataset],
        'summarize': [d['summarize'] for d in train_dataset]
    })
    valid_hf = Dataset.from_dict({
        'body': [d['body'] for d in valid_dataset],
        'summarize': [d['summarize'] for d in valid_dataset]
    })
    test_hf = Dataset.from_dict({
        'body': [d['body'] for d in test_dataset],
        'summarize': [d['summarize'] for d in test_dataset]
    })
    
    # 전처리 적용
    print("  [데이터 전처리 적용 중...]")
    train_dataset_processed = train_hf.map(
        preprocess_function,
        batched=True,
        remove_columns=train_hf.column_names
    )
    valid_dataset_processed = valid_hf.map(
        preprocess_function,
        batched=True,
        remove_columns=valid_hf.column_names
    )
    test_dataset_processed = test_hf.map(
        preprocess_function,
        batched=True,
        remove_columns=test_hf.column_names
    )
    
    # Data Collator 설정
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # ROUGE 메트릭 설정 (요약 품질 평가)
    rouge_metric = evaluate.load('rouge')
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        
        # 디코딩
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # 레이블에서 -100을 제거 (패딩 토큰)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # ROUGE 점수 계산
        result = rouge_metric.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True
        )
        
        return {
            'rouge1': result['rouge1'],
            'rouge2': result['rouge2'],
            'rougeL': result['rougeL']
        }
    
    # Early Stopping Callback 설정
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=3,        # 3 에폭 동안 개선 없으면 종료
        early_stopping_threshold=0.0001   # 최소 개선 임계값
    )
    
    # Trainer 초기화
    print("\n  [Trainer 초기화...]")
    print("  - Early Stopping: 활성화")
    print("  - Patience: 3 에폭")
    print("  - Threshold: 0.0001")
    print("  ✨ 검증 손실이 3 에폭 연속 개선되지 않으면 자동 종료")
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_processed,
        eval_dataset=valid_dataset_processed,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback]   # Early Stopping 추가
    )
    
    """
    4. KoBART 모델 훈련 및 평가
    """
    print("\n[4단계] 모델 훈련 시작...")
    print("-"*50)
    
    # 훈련 시작
    train_result = trainer.train()
    
    print("\n[훈련 완료]")
    print(f"  - 최종 Loss: {train_result.training_loss:.4f}")
    print(f"  - 실제 훈련된 에폭: {int(train_result.metrics['epoch'])} / 50")
    if int(train_result.metrics['epoch']) < 50:
        print(f"  ✅ Early Stopping 작동: 과적합 방지 성공!")
    else:
        print(f"  - 최대 에폭까지 훈련 완료")
    
    # 모델 저장
    print("\n[모델 저장 중...]")
    trainer.save_model('./kobart_final_model')
    tokenizer.save_pretrained('./kobart_final_model')
    print("  - 모델 저장 완료: ./kobart_final_model")
    
    # 검증 데이터셋 평가
    print("\n[검증 데이터셋 평가 중...]")
    eval_result = trainer.evaluate()
    print(f"  - Validation Loss: {eval_result['eval_loss']:.4f}")
    print(f"  - ROUGE-1: {eval_result['eval_rouge1']:.4f}")
    print(f"  - ROUGE-2: {eval_result['eval_rouge2']:.4f}")
    print(f"  - ROUGE-L: {eval_result['eval_rougeL']:.4f}")
    
    # 테스트 데이터셋 평가
    print("\n[테스트 데이터셋 평가 중...]")
    test_results = trainer.predict(test_dataset_processed)
    test_metrics = test_results.metrics
    print(f"  - Test Loss: {test_metrics['test_loss']:.4f}")
    print(f"  - Test ROUGE-1: {test_metrics['test_rouge1']:.4f}")
    print(f"  - Test ROUGE-2: {test_metrics['test_rouge2']:.4f}")
    print(f"  - Test ROUGE-L: {test_metrics['test_rougeL']:.4f}")
    
    # 평가 결과 저장
    with open('./evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump({
            'validation': eval_result,
            'test': test_metrics
        }, f, ensure_ascii=False, indent=2)
    print("\n  - 평가 결과 저장 완료: ./evaluation_results.json")
    
    return trainer, tokenizer, model


def inference_example(trainer=None, tokenizer=None, model=None):
    """
    5. KoBART 모델 추론
    전처리되지 않은 본문으로 추론 예시
    """
    print("\n" + "="*50)
    print("KoBART 문서 요약 모델 추론")
    print("="*50)
    
    # 모델이 제공되지 않은 경우 로드
    if model is None or tokenizer is None:
        print("\n[저장된 모델 로드 중...]")
        model_path = './kobart_final_model'
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        print(f"  - 모델 로드 완료 (디바이스: {device})")
    
    # 전처리되지 않은 예시 본문 (기자명, 이메일 포함)
    raw_text_lines = [
        "김철수 기자 = 인공지능 기술이 급격히 발전하면서 다양한 산업 분야에서 활용되고 있다.",
        "특히 자연어 처리 분야에서는 BERT, GPT와 같은 대규모 언어 모델이 등장했다.",
        "이메일: reporter@example.com",
        "한국에서도 SK텔레콤이 KoBART, KoBERT 등 한국어 특화 모델을 개발했다.",
        "이러한 모델들은 문서 요약, 감성 분석, 질의응답 등에 활용되고 있다.",
        "김영희 기자(younghee@news.com)는 이러한 기술이 미디어 산업에도 큰 영향을 미칠 것으로 전망했다.",
        "앞으로 인공지능 기술은 더욱 정교해질 것으로 예상된다."
    ]
    
    print("\n[원본 텍스트]")
    for i, line in enumerate(raw_text_lines, 1):
        print(f"  {i}. {line}")
    
    # 전처리 파이프라인 적용
    print("\n[전처리 시작]")
    print("  1. 기자명 및 이메일 제거...")
    filtered_lines = filter_text(raw_text_lines)
    print(f"     -> {len(raw_text_lines) - len(filtered_lines)}개 문장 제거됨")
    
    print("  2. 형태소 분석 및 불용어 제거...")
    processed_lines = preprocess_text_for_inference(raw_text_lines)
    
    print("\n[전처리된 텍스트]")
    for i, line in enumerate(processed_lines, 1):
        print(f"  {i}. {line}")

    # 전처리된 텍스트를 하나의 문자열로 결합
    input_text = ' '.join(processed_lines)
    
    print("\n[토크나이징]")
    # 토크나이징
    inputs = tokenizer(
        input_text,
        max_length=1024,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    
    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    print(f"  - 입력 토큰 길이: {inputs['input_ids'].shape[1]}")
    
    print("\n[모델 추론 중...]")
    # 모델 추론
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            max_length=256,              # 요약문 최대 길이
            num_beams=5,                 # Beam search
            early_stopping=True,
            no_repeat_ngram_size=2,      # 반복 방지
            length_penalty=1.0,
            temperature=1.0
        )
    
    # 디코딩
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("\n" + "="*50)
    print("[생성된 요약문]")
    print("="*50)
    print(summary)
    print("="*50)
    
    # 추가 예시들
    print("\n\n[추가 예시 1: 뉴스 기사]")
    news_text = [
        "박지성 기자 = 서울시가 2025년 스마트시티 프로젝트를 본격 추진한다.",
        "이번 프로젝트는 총 5000억원의 예산이 투입된다.",
        "인공지능, IoT, 빅데이터 기술을 활용해 교통, 환경, 안전 분야를 개선할 계획이다.",
        "연락처: park@seoul.go.kr",
        "시민들의 삶의 질 향상이 기대된다."
    ]
    
    processed = preprocess_text_for_inference(news_text)
    
    input_text = ' '.join(processed)
    inputs = tokenizer(input_text, max_length=1024, truncation=True, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            max_length=256,
            num_beams=5,
            early_stopping=True
        )
    
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"원문: {' '.join(news_text)}")
    print(f"\n요약: {summary}")
    
    print("\n\n[추가 예시 2: 기술 문서]")
    tech_text = [
        "이재용 기자 = 딥러닝은 인공신경망을 기반으로 한 기계학습 방법이다.",
        "다층 신경망 구조를 통해 복잡한 패턴을 학습할 수 있다.",
        "이미지 인식, 음성 인식, 자연어 처리 등에 활용된다.",
        "문의: tech@ai.com",
        "최근에는 Transformer 구조가 주목받고 있다."
    ]
    
    processed = preprocess_text_for_inference(tech_text)
    input_text = ' '.join(processed)
    inputs = tokenizer(input_text, max_length=1024, truncation=True, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            max_length=256,
            num_beams=5,
            early_stopping=True
        )
    
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"원문: {' '.join(tech_text)}")
    print(f"\n요약: {summary}")


def main():
    """메인 실행 함수 - 모델 훈련 후 바로 추론 수행"""
    
    print("\n" + "="*70)
    print(" "*15 + "KoBART 문서 요약 모델 훈련 및 추론")
    print("="*70)
    
    # 모델 훈련 후 바로 추론 실행
    print("\n[자동 실행 모드: 모델 훈련 → 추론]")
    print("-"*70)
    
    # 모델 훈련
    trainer, tokenizer, model = train_model()
    
    # 훈련 완료 후 추론
    inference_example(trainer, tokenizer, model)
    
    print("\n" + "="*70)
    print(" "*25 + "프로그램 종료")
    print("="*70)


if __name__ == "__main__":
    main()
