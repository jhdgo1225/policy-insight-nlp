# Ollama 환경 구축 가이드

## 1단계: Ollama 설치

### macOS (Homebrew 사용)

```bash
# Homebrew로 설치 (권장)
brew install ollama

# 또는 공식 인스톨러 다운로드
# https://ollama.ai/download 에서 macOS용 다운로드
```

### 수동 설치

```bash
# 공식 설치 스크립트
curl -fsSL https://ollama.ai/install.sh | sh
```

## 2단계: Ollama 서비스 시작

```bash
# Ollama 서버 백그라운드 실행
ollama serve

# 또는 별도 터미널에서 실행하여 로그 확인
```

**중요**: Ollama는 기본적으로 `http://localhost:11434`에서 실행됩니다.

## 3단계: 한국어 지원 모델 다운로드

### 추천 모델 (한국어 성능 좋음)

```bash
# 1. Llama 3 (8B) - 영어/한국어 균형
ollama pull llama3:8b

# 2. Gemma 2 (9B) - 가볍고 빠름
ollama pull gemma2:9b

# 3. Qwen 2.5 (7B) - 한국어 성능 우수
ollama pull qwen2.5:7b

# 4. EEVE (한국어 특화 - 권장!)
ollama pull heegyu/EEVE-Korean-Instruct-10.8B-v1.0:latest

# 5. 경량 모델 (빠른 테스트용)
ollama pull llama3:3b
```

### 모델 확인

```bash
# 다운로드된 모델 목록 확인
ollama list

# 모델 테스트
ollama run llama3:8b "안녕하세요. 법령에 대해 설명해주세요."
```

## 4단계: Python 라이브러리 설치

```bash
# Ollama와 LangChain 통합 라이브러리
pip install langchain-ollama

# 또는 기존 설치에 추가
pip install langchain-community
```

## 5단계: 환경 변수 설정 (선택사항)

```bash
# ~/.zshrc 또는 ~/.bashrc에 추가
export OLLAMA_HOST="http://localhost:11434"
export OLLAMA_NUM_PARALLEL=2  # 병렬 처리 수
export OLLAMA_MAX_LOADED_MODELS=1  # 메모리 절약
```

## 성능 비교

| 항목           | HuggingFace (CPU) | Ollama    |
| -------------- | ----------------- | --------- |
| 초기 로딩      | 1-3분             | 1-5초     |
| 추론 속도 (8B) | 30초-5분          | 2-10초    |
| 메모리 사용량  | ~16GB             | ~8GB      |
| GPU 지원       | 복잡함            | 자동 감지 |

## Ollama 장점

1. **빠른 추론**: 최적화된 C++ 엔진
2. **간편한 모델 관리**: `ollama pull/list/rm`
3. **자동 GPU 감지**: CUDA, Metal(Apple Silicon) 자동 활용
4. **메모리 효율**: 양자화 모델 (4bit/8bit)
5. **REST API**: 외부 애플리케이션 통합 용이

## 모델 선택 가이드

### 법령 예측용 추천 모델

1. **Qwen 2.5 (7B)** - 최고 추천

   - 한국어 성능 우수
   - 법률/정책 문서 이해 뛰어남
   - 빠른 속도

2. **Llama 3 (8B)**

   - 범용성 우수
   - 안정적인 성능
   - 커뮤니티 지원 많음

3. **Gemma 2 (9B)**

   - Google 모델
   - 빠른 추론 속도
   - 긴 컨텍스트 지원

4. **EEVE (10.8B)** - 한국어 특화
   - 한국어 전용 파인튜닝
   - 법령/정책 문서 특화
   - 약간 느릴 수 있음

## 문제 해결

### Ollama 서비스가 실행되지 않을 때

```bash
# 프로세스 확인
ps aux | grep ollama

# 강제 종료 후 재시작
pkill ollama
ollama serve
```

### 포트 충돌 시

```bash
# 다른 포트로 실행
OLLAMA_HOST=0.0.0.0:11435 ollama serve
```

### 메모리 부족 시

```bash
# 더 작은 모델 사용
ollama pull llama3:3b
ollama pull gemma2:2b
```

## 다음 단계

1. ✅ Ollama 설치 완료
2. ✅ 모델 다운로드 완료
3. ✅ `ollama serve` 실행 중
4. → `script_ollama.py` 실행
