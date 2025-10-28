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


if __name__ == "__main__":
	id2label = {
		0: "부정",
		1: "중립",
		2: "긍정"
	}

	print("\n=== Inference Examples ===")
	test_sentences = [
		"이 주식은 정말 좋은 투자 기회인 것 같습니다.",
		"시장 상황이 불안정하여 걱정됩니다.",
		"오늘 주가는 보합세를 보이고 있습니다."
	]

	for sentence in test_sentences:
		pred_class, confidence, probs = predict_sentiment(sentence, model, tokenizer, device)
		print(f"\nText: {sentence}")
		print(f"Predicted Class: {id2label[pred_class]}")
		print(f"Confidence: {confidence:.4f}")
		print(f"Probabilities: {probs}")

	print("\n=== Inference finished ===")