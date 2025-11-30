"""
KoBART 모델 추론 전용 스크립트
저장된 모델을 로드하여 텍스트 요약을 수행합니다.
"""
import json
import os
import requests
import re
import torch
import time
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# tokenizers 병렬 처리 충돌 방지
os.environ["TOKENIZERS_PARALLELISM"] = "false"


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
    """
    # 1. 기자/이메일 제거
    filtered_lines = filter_text(text_lines)

    return filtered_lines


def summarize_text(text_input, model, tokenizer, device):
    """
    텍스트를 요약하는 함수

    Args:
        text_input: 입력 텍스트 (문자열 또는 문자열 리스트)
        model: 로드된 KoBART 모델
        tokenizer: 토크나이저
        device: 디바이스 (cuda/cpu)

    Returns:
        요약문 (문자열)
    """
    # 리스트인 경우 전처리 수행
    if isinstance(text_input, list):
        processed_lines = preprocess_text_for_inference(text_input)
        # 요약 태스크를 인식하기 위해 전처리 문장 리스트의 첫 번째 요소로 "요약: " 추가
        input_text = ' '.join(processed_lines)
    else:
        input_text = text_input

    # 토크나이징
    inputs = tokenizer(
        input_text,
        max_length=1024,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 모델 추론
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            max_length=256,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=2,
            length_penalty=1.0,
            temperature=1.0
        )

    # 디코딩
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary


def main():
    """메인 실행 함수"""
    print("\n")
    print(" "*20 + "KoBART 텍스트 요약 (추론 전용)")
    print("="*70)

    with open("./newspaper_summarize_jsonl/newspaper_summarize.jsonl") as f:
        datasets = [json.loads(line) for line in f]
    model_path = './kobart_final_model'

    if not os.path.exists(model_path):
        print(f"❌ 오류: 모델을 찾을 수 없습니다: {model_path}")
        print("   먼저 'kobart_summarization_complete.py'를 실행하여 모델을 훈련하세요.")
        return

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()  # 평가 모드
        print(f"✅ 모델 로드 완료 (디바이스: {device})")
    except Exception as e:
        print(f"❌ 모델 로드 중 오류 발생: {e}")
        return

    start_time = time.time()
    pred = summarize_text(datasets[0]['body'], model, tokenizer, device)
    end_time = time.time()
    print(f"모델 추론 시간: {(end_time - start_time):.3f}초")
    result = datasets[0]['summarize']
    print("[본문]")
    print(" ".join(datasets[0]['body']))
    print(f"[실제 요약문]: {result}")
    print(f"[예상 요약문]: {pred}")

    print("="*70)
    print(" "*25 + "프로그램 종료")


if __name__ == "__main__":
    main()