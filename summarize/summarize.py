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
    """기자/이메일 패턴을 빈 문자열로 치환"""
    reporter_pattern = r'[가-힣]{2,4}\s+기자'
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    name_email_pattern = r'[가-힣]{2,4}\s+[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'

    def remove_patterns(segment):
        """기자/이메일 패턴을 빈 문자열로 치환"""
        segment = re.sub(name_email_pattern, '', segment)  # 이름+이메일 먼저 제거
        segment = re.sub(reporter_pattern, '', segment)
        segment = re.sub(email_pattern, '', segment)
        return segment.strip()

    return [remove_patterns(line) for line in lines]


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

def load_datasets():
    """newspaper_summarize.jsonl 파일을 로드하고 80:10:10 비율로 분리"""
    # JSONL 파일 읽기
    dataset = []
    with open('./newspaper_summarize_jsonl/newspaper_summarize.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            dataset.append(data)
    
    print(f"  - 전체 데이터: {len(dataset)}개")
    
    # 첫 번째 split: 80% train, 20% temp (valid + test)
    train_dataset, temp_dataset = train_test_split(
        dataset,
        test_size=0.2,
        random_state=42
    )
    
    # 두 번째 split: temp를 50:50으로 나누어 valid와 test 생성 (각각 10%)
    valid_dataset, test_dataset = train_test_split(
        temp_dataset,
        test_size=0.5,
        random_state=42
    )
    
    print(f"  - Train: {len(train_dataset)}개 ({len(train_dataset)/len(dataset)*100:.1f}%)")
    print(f"  - Valid: {len(valid_dataset)}개 ({len(valid_dataset)/len(dataset)*100:.1f}%)")
    print(f"  - Test: {len(test_dataset)}개 ({len(test_dataset)/len(dataset)*100:.1f}%)")
    
    # "요약: " 프리픽스 추가
    train_dataset = list(map(func, train_dataset))
    valid_dataset = list(map(func, valid_dataset))
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
    train_dataset, valid_dataset, test_dataset = load_datasets()
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
    print("  - Max epochs: 3")
    print("  - Max input length: 512")
    print("  - Max summary length: 128")
    print("  - Gradient accumulation: 2")
    print("  - Warmup steps: 500")
    
    training_args = Seq2SeqTrainingArguments(
        output_dir='./kobart_summarization_results',
        num_train_epochs=3,                    # 3 에폭으로 제한
        per_device_train_batch_size=2,         # GPU 메모리에 따라 조정 (8-16)
        per_device_eval_batch_size=2,
        learning_rate=3e-5,                    # KoBART 문서 요약에서 가장 일반적인 값
        weight_decay=0.01,                     # 가중치 감쇠
        warmup_steps=500,                      # 학습률 워밍업
        logging_dir='./logs',
        logging_steps=100,
        save_strategy="epoch",                 # 에폭마다 저장 (eval_strategy와 일치)
        save_total_limit=3,                    # 최대 체크포인트 개수 (최근 3개만 유지)
        eval_strategy="epoch",                 # 에폭마다 평가
        load_best_model_at_end=False,          # 최적 모델 자동 로드 비활성화
        metric_for_best_model="eval_loss",     # 검증 손실 기준
        greater_is_better=False,               # 손실은 낮을수록 좋음
        predict_with_generate=True,            # 생성 기반 평가
        generation_max_length=128,             # 요약문 최대 길이
        gradient_accumulation_steps=2,         # 배치 크기를 효과적으로 늘림
        fp16=torch.cuda.is_available(),        # Mixed precision (GPU에서만)
        report_to="none",                      # wandb 등 로깅 비활성화
    )
    
    # 데이터 전처리: body를 입력으로, summarize를 레이블로
    def preprocess_function(examples):
        # body는 문장 리스트이므로 join
        inputs = [' '.join(body) if isinstance(body, list) else body for body in filter_text(examples['body'])]
        targets = examples['summarize']
        
        # 토크나이징
        model_inputs = tokenizer(
            inputs, 
            max_length=512,          # 입력 최대 길이
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # 레이블 토크나이징
        labels = tokenizer(
            targets,
            max_length=128,           # 요약문 최대 길이
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
    
    # Trainer 초기화
    print("\n  [Trainer 초기화...]")
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_processed,
        eval_dataset=valid_dataset_processed,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
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
    print(f"  - 훈련된 에폭: {int(train_result.metrics['epoch'])} / 3")
    
    # 모델 저장
    print("\n[모델 저장 중...]")
    trainer.save_model('./kobart_final_model_v3')
    tokenizer.save_pretrained('./kobart_final_model_v3')
    print("  - 모델 저장 완료: ./kobart_final_model_v3")
    
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
    
    print("\n" + "="*70)
    print(" "*25 + "프로그램 종료")
    print("="*70)


if __name__ == "__main__":
    main()
