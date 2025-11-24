from transformers import BertForSequenceClassification, BertTokenizer
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import re
import requests
from konlpy.tag import Okt
from tqdm import tqdm
import os

# ========================================
# 1. 설정 및 초기화
# ========================================
model_name_or_path = "jinmang2/kpfbert"
output_dir = "./kpfbert_sentiment_model"
num_labels = 3  # 긍정/중립/부정 (데이터에 맞게 조정)

# 디바이스 설정
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    # MPS는 학습에만 사용, 추론은 CPU 사용 권장
    device = torch.device('mps')
    print("Warning: Using MPS device. Inference will fall back to CPU for stability.")
else:
    device = torch.device('cpu')
print(f"Using device: {device}")

# ========================================
# 2. 데이터 로드 및 전처리
# ========================================
# kpfbert 모델 및 토크나이저 로드 후 약 3GB 소요

# 불용어 리스트 다운로드
url = "https://raw.githubusercontent.com/stopwords-iso/stopwords-ko/master/stopwords-ko.txt"
korean_stopwords = set(requests.get(url).text.split("\n"))
for stopword in ["은", "는", "이", "가", "도", "고", "정말", "의", "이나", "이라고", "인", "이다", "하여", "``", "에", "에는"]:
	korean_stopwords.add(stopword) 
korean_stopwords.remove("모")

# 데이터 전처리
okt = Okt()

def clean_text(text):
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

# 데이터셋 로드
dataset = load_dataset('csv', data_files='finance_data.csv', encoding='utf-8')

# 컬럼 정리 (원본 코드에서 제공된 형식 기준)
if 'sentence' in dataset['train'].column_names:
	dataset['train'] = dataset['train'].remove_columns('sentence')
if 'kor_sentence' in dataset['train'].column_names:
	dataset = dataset.rename_column('kor_sentence', 'sentence')

dataset = dataset['train'].train_test_split(test_size=0.2)
dataset_val_test = dataset['test'].train_test_split(test_size=0.5)
dataset['valid'] = dataset_val_test['train']
dataset['test'] = dataset_val_test['test']

label2id = {
	'negative': 0,
	'neutral': 1,
	'positive': 2
}

def convert_label(example):
	example['labels'] = label2id[example['labels']]
	return example

dataset = dataset.map(convert_label)

# ========================================
# 3. 토크나이저 및 데이터 전처리
# ========================================

# 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained(model_name_or_path)

def preprocess_function(examples):
	"""데이터셋 토크나이징 함수"""
	# 텍스트 정제
	cleaned_texts = [clean_text(sentence) for sentence in examples['sentence']]

	# 토크나이징
	tokenized = tokenizer(
		cleaned_texts,
		padding=False,  # DataCollator가 배치별로 처리
		truncation=True,
		max_length=128,
		return_tensors=None
	)

	return tokenized

tokenized_dataset = dataset.map(
	preprocess_function,
	batched=True,
	remove_columns=['sentence'],
	desc="Tokenizing"
)

# 레이블 컬럼명 확인 및 변경 (label이 아닌 경우)
if 'labels' not in tokenized_dataset['train'].column_names:
    # 실제 레이블 컬럼명에 맞게 수정 (예: 'sentiment', 'category' 등)
    label_column = [col for col in tokenized_dataset['train'].column_names 
                   if col not in ['input_ids', 'token_type_ids', 'attention_mask']][0]
    tokenized_dataset = tokenized_dataset.rename_column(label_column, 'labels')

# 데이터 포맷 설정
tokenized_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])


# ========================================
# 4. 모델 로드 및 설정
# ========================================

# 분류 모델 로드
model = BertForSequenceClassification.from_pretrained(
    model_name_or_path,
    num_labels=num_labels,
    problem_type="single_label_classification"
)

model.to(device)

# Data Collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ========================================
# 5. 평가 메트릭 정의
# ========================================

def compute_metrics(eval_pred):
    """평가 메트릭 계산"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    # 정확도
    accuracy = accuracy_score(labels, predictions)

    # Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# ========================================
# 6. 학습 설정
# ========================================

training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_dir='./logs',
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    greater_is_better=True,
    save_total_limit=2,
    seed=42,
    fp16=torch.cuda.is_available(),  # GPU 사용 시 mixed precision
)

# ========================================
# 7. Trainer 초기화 및 학습
# ========================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['valid'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 학습 시작
print("Starting training...")
train_result = trainer.train()

# 학습 결과 저장
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

# 학습 메트릭 출력
print("\n=== Training Results ===")
print(train_result)

# ========================================
# 8. 검증 세트 평가
# ========================================

print("\n=== Validation Results ===")
eval_results = trainer.evaluate(tokenized_dataset['valid'])
print(eval_results)

# ========================================
# 9. 테스트 세트 평가
# ========================================

print("\n=== Test Results ===")
test_results = trainer.predict(tokenized_dataset['test'])
print(f"Test Accuracy: {test_results.metrics['test_accuracy']:.4f}")
print(f"Test F1 Score: {test_results.metrics['test_f1']:.4f}")

# 상세 분류 리포트
test_predictions = np.argmax(test_results.predictions, axis=1)
test_labels = test_results.label_ids

print("\n=== Classification Report ===")
print(classification_report(
    test_labels, 
    test_predictions,
    target_names=['Label_0', 'Label_1', 'Label_2']  # 실제 레이블명으로 변경
))

# ========================================
# 10. 추론 함수 정의
# ========================================

def predict_sentiment(text, model, tokenizer, device):
    """단일 텍스트에 대한 감정 예측"""
    # 모델을 평가 모드로 설정
    model.eval()

	# MPS 장치 사용 시 CPU로 폴백 (안정성)
    inference_device = device
    if device.type == 'mps':
        print("Warning: MPS device detected. Switching to CPU for inference to avoid MPS errors.")
        inference_device = torch.device('cpu')
        model.to(inference_device)

    # 텍스트 전처리
    cleaned_text = clean_text(text)

    # 토크나이징
    inputs = tokenizer(
        cleaned_text,
        return_tensors='pt',
        truncation=True,
        max_length=128,
        padding=True
    )

    # 디바이스로 이동
    inputs = {k: v.to(inference_device) for k, v in inputs.items()}

    # 예측
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()
        confidence = predictions[0][predicted_class].item()

    return predicted_class, confidence, predictions[0].cpu().numpy()

# ========================================
# 11. 추론 예시
# ========================================

print("\n=== Inference Examples ===")
test_sentences = [
    "이 주식은 정말 좋은 투자 기회인 것 같습니다.",
    "시장 상황이 불안정하여 걱정됩니다.",
    "오늘 주가는 보합세를 보이고 있습니다."
]

for sentence in test_sentences:
    pred_class, confidence, probs = predict_sentiment(sentence, model, tokenizer, device)
    print(f"\nText: {sentence}")
    print(f"Predicted Class: {pred_class}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Probabilities: {probs}")

print("\n=== Training Complete! ===")
print(f"Model saved to: {output_dir}")