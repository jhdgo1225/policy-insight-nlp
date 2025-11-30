#!/bin/bash
# Ollama 기반 RAG 시스템 실행 스크립트

echo "======================================"
echo "법령 예측 RAG 시스템 (Ollama)"
echo "======================================"

# 1. Ollama 서비스 확인
echo -e "\n[1/4] Ollama 서비스 확인 중..."
if ! pgrep -x "ollama" > /dev/null; then
    echo "❌ Ollama가 실행되지 않았습니다."
    echo "터미널에서 'ollama serve'를 실행해주세요."
    echo ""
    echo "새 터미널 창에서 실행:"
    echo "  ollama serve"
    exit 1
fi
echo "✓ Ollama 서비스 실행 중"

# 2. 모델 다운로드 확인
echo -e "\n[2/4] 모델 확인 중..."
OLLAMA_MODEL="qwen2.5:7b"

if ! ollama list | grep -q "$OLLAMA_MODEL"; then
    echo "⚠️  모델이 설치되지 않았습니다: $OLLAMA_MODEL"
    echo "모델 다운로드를 시작합니다..."
    ollama pull "$OLLAMA_MODEL"
    
    if [ $? -eq 0 ]; then
        echo "✓ 모델 다운로드 완료"
    else
        echo "❌ 모델 다운로드 실패"
        exit 1
    fi
else
    echo "✓ 모델이 이미 설치되어 있습니다: $OLLAMA_MODEL"
fi

# 3. Python 환경 확인
echo -e "\n[3/4] Python 환경 확인 중..."
if ! python -c "import langchain_ollama" 2>/dev/null; then
    echo "⚠️  langchain-ollama 패키지가 설치되지 않았습니다."
    echo "패키지를 설치합니다..."
    pip install langchain-ollama
fi
echo "✓ Python 환경 준비 완료"

# 4. 스크립트 실행
echo -e "\n[4/4] RAG 시스템 실행 중..."
echo "======================================"
cd "$(dirname "$0")"
python script_ollama.py

echo -e "\n======================================"
echo "실행 완료"
echo "======================================"
