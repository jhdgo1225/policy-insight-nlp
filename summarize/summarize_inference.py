"""
KoBART 모델 추론 전용 스크립트
저장된 모델을 로드하여 텍스트 요약을 수행합니다.
"""
import json
import os
import requests
import re
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# tokenizers 병렬 처리 충돌 방지
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 전역 변수
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


def filter_text(lines):
    """기자/이메일 포함 문장 제거"""
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
    
    return processed_lines


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
    print("\n" + "="*70)
    print(" "*20 + "KoBART 텍스트 요약 (추론 전용)")
    print("="*70)
    
    # 모델 로드
    print("\n[1단계] 저장된 모델 로드 중...")
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
    
    # 예시 텍스트로 추론
    print("\n[2단계] 예시 텍스트로 요약 수행\n")
    
    # 예시 1: 뉴스 기사
    print("-"*70)
    print("[예시 1] 뉴스 기사")
    print("-"*70)
    news_text = [
        "김철수 기자 = 인공지능 기술이 급격히 발전하면서 다양한 산업 분야에서 활용되고 있다.",
        "특히 자연어 처리 분야에서는 BERT, GPT와 같은 대규모 언어 모델이 등장했다.",
        "이메일: reporter@example.com",
        "한국에서도 SK텔레콤이 KoBART, KoBERT 등 한국어 특화 모델을 개발했다.",
        "이러한 모델들은 문서 요약, 감성 분석, 질의응답 등에 활용되고 있다.",
        "김영희 기자(younghee@news.com)는 이러한 기술이 미디어 산업에도 큰 영향을 미칠 것으로 전망했다.",
        "앞으로 인공지능 기술은 더욱 정교해질 것으로 예상된다."
    ]
    
    print("\n[원본]")
    for i, line in enumerate(news_text, 1):
        print(f"  {i}. {line}")
    
    summary = summarize_text(news_text, model, tokenizer, device)
    print(f"\n[요약문]\n  {summary}\n")
    
    # 예시 2: 기술 문서
    print("-"*70)
    print("[예시 2] 기술 문서")
    print("-"*70)
    tech_text = [
        "박지성 기자 = 서울시가 2025년 스마트시티 프로젝트를 본격 추진한다.",
        "이번 프로젝트는 총 5000억원의 예산이 투입된다.",
        "인공지능, IoT, 빅데이터 기술을 활용해 교통, 환경, 안전 분야를 개선할 계획이다.",
        "연락처: park@seoul.go.kr",
        "시민들의 삶의 질 향상이 기대된다."
    ]
    
    print("\n[원본]")
    for i, line in enumerate(tech_text, 1):
        print(f"  {i}. {line}")
    
    summary = summarize_text(tech_text, model, tokenizer, device)
    print(f"\n[요약문]\n  {summary}\n")
    
    # 예시 3: 간단한 문장
    print("-"*70)
    print("[예시 3] 정제된 텍스트 (전처리 없이 바로 요약)")
    print("-"*70)
    simple_text = "딥러닝은 인공신경망을 기반으로 한 기계학습 방법이다. 다층 신경망 구조를 통해 복잡한 패턴을 학습할 수 있다. 이미지 인식, 음성 인식, 자연어 처리 등에 활용된다."
    
    print(f"\n[원본]\n  {simple_text}")
    
    summary = summarize_text(simple_text, model, tokenizer, device)
    print(f"\n[요약문]\n  {summary}\n")
    
    print("="*70)
    
    # 사용자 정의 텍스트 입력 옵션
    print("\n[3단계] 직접 텍스트 입력하기")
    print("-"*70)
    print("직접 텍스트를 입력하여 요약을 생성할 수 있습니다.")
    print("여러 줄을 입력하려면 각 줄을 입력 후 Enter를 누르세요.")
    print("입력을 마치려면 빈 줄에서 Enter를 누르세요.")
    print("(건너뛰려면 바로 Enter를 누르세요)\n")
    
    user_lines = []
    print("텍스트 입력:")
    while True:
        line = input()
        if line.strip() == "":
            break
        user_lines.append(line)
    
    if user_lines:
        print("\n[입력한 텍스트]")
        for i, line in enumerate(user_lines, 1):
            print(f"  {i}. {line}")
        
        summary = summarize_text(user_lines, model, tokenizer, device)
        print(f"\n[생성된 요약문]\n  {summary}\n")
    else:
        print("\n입력이 없어 건너뜁니다.\n")
    
    print("="*70)
    print(" "*25 + "프로그램 종료")
    print("="*70)
    print("\n💡 Tip: 이 스크립트를 수정하여 파일에서 텍스트를 읽거나")
    print("        API 서버로 만들어 활용할 수 있습니다!")


if __name__ == "__main__":
    main()