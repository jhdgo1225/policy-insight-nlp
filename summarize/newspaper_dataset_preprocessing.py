import json
import os
import asyncio
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# tokenizers 병렬 처리 충돌 방지
os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_ckpt = 'gogamza/kobart-base-v2'
# output_dir = "./kobart_korean_summarize_model"

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 토큰 길이 제한 설정
MAX_BODY_LENGTH = 512
MAX_SUMMARY_LENGTH = 128

tokenizer = AutoTokenizer.from_pretrained(model_ckpt, force_download=True)

# tokenizer 접근을 위한 락 생성 (공유 자원 보호)
tokenizer_lock = asyncio.Lock()

async def extract_summarize_and_body(article):
	"""비동기적으로 기사에서 요약문과 본문을 추출하고 토큰 길이 제한을 검증합니다."""
	summarize = article['abstractive'][0]
	body = []
	if article['text']:
		# 변수명 수정: one_article_info -> article
		body = [text['sentence'] for texts in article['text'] for text in texts]
	
	# 본문과 요약문을 문자열로 결합
	body_text = ' '.join(body)
	
	# 토큰화 및 길이 검증 (공유 자원인 tokenizer 사용 - 락으로 보호)
	async with tokenizer_lock:
		body_tokens = tokenizer.encode(body_text, add_special_tokens=True, truncation=False)
		summary_tokens = tokenizer.encode(summarize, add_special_tokens=True, truncation=False)
	
	# 토큰 길이 확인
	body_token_length = len(body_tokens)
	summary_token_length = len(summary_tokens)
	
	# 토큰 길이 제한 검증 및 반환
	if body_token_length > MAX_BODY_LENGTH or summary_token_length > MAX_SUMMARY_LENGTH:
		# 제한 초과 시 빈 딕셔너리 반환
		return {}
	else:
		# 제한 이하 시 텍스트 상태로 반환
		return {
			'body': body_text,
			'summarize': summarize
		}
async def process_batch(batch, semaphore):
	"""세마포어를 사용해 동시 실행 수를 제한하면서 배치를 처리합니다."""
	async with semaphore:
		tasks = [extract_summarize_and_body(article) for article in batch]
		return await asyncio.gather(*tasks)

async def original_to_jsonl_async(dataset, batch_size=100, max_concurrent=10):
	"""비동기 방식으로 데이터셋을 JSONL 형태로 변환하고 토큰 길이 필터링을 적용합니다."""
	semaphore = asyncio.Semaphore(max_concurrent)
	
	# 데이터를 배치로 나누기
	batches = [dataset[i:i + batch_size] for i in range(0, len(dataset), batch_size)]
	
	print(f"총 {len(batches)}개의 배치로 나누어 처리합니다 (배치 크기: {batch_size}, 최대 동시 실행: {max_concurrent})")
	print(f"토큰 길이 제한 - 본문: {MAX_BODY_LENGTH}, 요약문: {MAX_SUMMARY_LENGTH}")
	
	# 배치별로 비동기 처리
	all_results = []
	total_processed = 0
	total_valid = 0
	total_filtered = 0
	
	for i, batch in enumerate(batches):
		print(f"배치 {i+1}/{len(batches)} 처리 중...")
		batch_results = await process_batch(batch, semaphore)
		
		# 유효한 결과만 필터링 (빈 딕셔너리가 아닌 것만)
		valid_results = [r for r in batch_results if r]
		filtered_count = len(batch_results) - len(valid_results)
		
		all_results.extend(valid_results)
		total_processed += len(batch_results)
		total_valid += len(valid_results)
		total_filtered += filtered_count
		
		if filtered_count > 0:
			print(f"  - 필터링됨: {filtered_count}개 (토큰 길이 초과)")
	
	print(f"\n=== 처리 결과 ===")
	print(f"총 처리: {total_processed}개")
	print(f"유효: {total_valid}개 ({total_valid/total_processed*100:.2f}%)")
	print(f"필터링: {total_filtered}개 ({total_filtered/total_processed*100:.2f}%)")
	
	return all_results

def original_to_jsonl(dataset):
	"""동기 방식으로 데이터셋을 JSONL 형태로 변환합니다 (호환성 유지)."""
	return asyncio.run(original_to_jsonl_async(dataset))

async def main():
	"""메인 비동기 함수"""
	print("비동기 병렬 처리로 데이터 변환을 시작합니다...")
	train_dataset = load_dataset('json', data_files="./newspaper_summarize/train_original.json")
	test_dataset = load_dataset('json', data_files="./newspaper_summarize/valid_original.json")
	train_dataset = train_dataset['train']['documents']
	test_dataset = test_dataset['train']['documents']
	
	# 병렬로 훈련 및 테스트 데이터 처리
	train_task = original_to_jsonl_async(train_dataset, batch_size=50, max_concurrent=5)
	test_task = original_to_jsonl_async(test_dataset, batch_size=50, max_concurrent=5)
	
	print("훈련 및 테스트 데이터를 병렬로 처리 중...")
	train_jsonl, test_jsonl = await asyncio.gather(train_task, test_task)
	
	print(f'\n원본 훈련 데이터셋 크기: {len(train_dataset)}, 본문요약문만 추출된 데이터셋 크기: {len(train_jsonl)}')
	if len(train_dataset) != len(train_jsonl):
		print("원본 훈련 데이터셋(train_original.json)의 크기와 본문 요약문만 추출된 데이터셋 크기가 일치하지 않습니다!")

	print(f'원본 테스트 데이터셋 크기: {len(test_dataset)}, 본문요약문만 추출된 데이터셋 크기: {len(test_jsonl)}')
	if len(test_dataset) != len(test_jsonl):
		print("원본 테스트 데이터셋(valid_original.json)의 크기와 본문 요약문만 추출된 데이터셋 크기가 일치하지 않습니다!")

	# 결합된 데이터셋 생성
	combined_dataset = train_jsonl + test_jsonl
	
	# 디렉토리 생성 및 파일 저장
	if not os.path.isdir('./newspaper_summarize_jsonl'):
		os.mkdir('./newspaper_summarize_jsonl')
	
	with open("./newspaper_summarize_jsonl/newspaper_summarize.jsonl", "w", encoding="utf-8") as f:
		for item in combined_dataset:
			json.dump(item, f, ensure_ascii=False)
			f.write('\n')
	
	print(f"\n총 {len(combined_dataset)}개의 유효한 아이템이 저장되었습니다.")
	print(f"총 {len(combined_dataset)}개의 아이템이 저장되었습니다.")

# 동기 방식 호환성을 위한 기존 코드 (주석 처리)
# train_jsonl = original_to_jsonl(train_dataset)
# test_jsonl = original_to_jsonl(test_dataset)

if __name__ == "__main__":
	# 비동기 메인 함수 실행
	asyncio.run(main())
