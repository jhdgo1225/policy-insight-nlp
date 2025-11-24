# ========================================
# 1. 패키지 임포트 및 모델, 토크나이저 초기화
# ========================================
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from konlpy.tag import Okt
import requests

def load_model_and_tokenizer(model_ckpt):
	model = BertForSequenceClassification.from_pretrained(model_ckpt)
	tokenizer = BertTokenizer.from_pretrained(model_ckpt)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model.to(device)
	return model, tokenizer, device


def clean_text(text):
	url = "https://raw.githubusercontent.com/stopwords-iso/stopwords-ko/master/stopwords-ko.txt"
	korean_stopwords = set(requests.get(url).text.split("\n"))
	for stopword in ["은", "는", "이", "가", "도", "고", "정말", "의", "이나", "이라고", "인", "이다", "하여", "``", "에", "에는"]:
		korean_stopwords.add(stopword) 
	korean_stopwords.remove("모")

	# 데이터 전처리
	okt = Okt()

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

def predict_sentiment(text):
	"""단일 텍스트에 대한 감정 예측"""
	model_ckpt = './kpfbert_sentiment_model'
	model, tokenizer, device = load_model_and_tokenizer(model_ckpt)
	model.eval()

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
	inputs = {k: v.to(device) for k, v in inputs.items()}

	# 예측: 분류 모델의 logits를 사용
	with torch.no_grad():
		outputs = model(**inputs)
		logits = outputs.logits
		predictions = torch.nn.functional.softmax(logits, dim=-1)
		predicted_class = torch.argmax(predictions, dim=-1).item()
		confidence = predictions[0][predicted_class].item()

	return predicted_class, confidence, predictions[0].cpu().numpy()

def compute_metrics(predictions, labels):
    """평가 메트릭 계산"""
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

if __name__ == "__main__":
	id2label = {
		0: "부정",
		1: "중립",
		2: "긍정"
	}

	label2id = {
		"negative": 0,
		"neutral": 1,
		"positive": 2
	}

	eng_label_to_kor = {
		"negative": "부정",
		"neutral": "중립",
		"positive": "긍정"
	}

	print("\n=== Inference Examples ===")
	news_data = pd.read_csv("./finance_data.csv", encoding="utf-8")
	
	# 레이블별 데이터 개수 확인
	print("\n=== Label Distribution ===")
	label_counts = news_data['labels'].value_counts()
	print("\nRaw counts:")
	print(label_counts)
	
	print("\nPercentage:")
	print(label_counts / len(news_data) * 100, "%")
	
	print("\nDetailed distribution:")
	for label, count in label_counts.items():
		percentage = (count / len(news_data) * 100)
		print(f"{eng_label_to_kor[label]}: {count} samples ({percentage:.2f}%)")
	
	print("\n" + "="*50)
	
	news_data = news_data[:485]
	preds = []
	real = []

	for idx, sentence in enumerate(news_data['kor_sentence']):
		pred_class, confidence, probs = predict_sentiment(sentence)
		preds.append(pred_class)
		real.append(label2id[news_data['labels'][idx]])
		print(f"\nText: {sentence}")
		print(f"Predicted Class: {id2label[pred_class]}")
		print(f"Result Class: {eng_label_to_kor[news_data['labels'][idx]]}")
		print(f"Confidence: {confidence:.4f}")
		print(f"Probabilities: {probs}")

	pprint.pprint(compute_metrics(preds, real))
	print("\n=== Inference finished ===")