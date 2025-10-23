from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5Config
from datasets import load_dataset
import numpy as np
import evaluate
import torch
from collections import defaultdict

rouge = evaluate.load("rouge")

def compute_metrics_for_test(predictions, references):    
    scores = rouge.compute(
        predictions=predictions,
        references=references,
        rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"],
        use_stemmer=True,
    )
    return scores

model_ckpt = './text_summarize_model'
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt)

domain_name = 'law_small'

domain_data = {
    'train': f'{domain_name}/train_small.jsonl',
    'valid': f'{domain_name}/valid_small.jsonl',
    'test': f'{domain_name}/test_small.jsonl'
}

test_dataset = load_dataset('json', data_files=domain_data, split='test')
total_test_dataset = len(test_dataset)

test_metrics = defaultdict(float)

f = open("./text_summarize_result.txt", 'w')
for idx, data in enumerate(test_dataset):
    if (idx >= 30): break
    data['text'][0] = "summarize: " + data['text'][0]
    inputs = ' '.join(data['text'])
    label = data['label']
    inputs_tokenized = tokenizer(inputs, padding='max_length', max_length=2048, truncation=True, return_tensors='pt')
    inputs_ids = inputs_tokenized['input_ids']
    attention_mask = inputs_tokenized['attention_mask']
    label_ids = tokenizer(label, padding='max_length', max_length=512, truncation=True, return_tensors='pt').input_ids
    with torch.no_grad():
        generation = model.generate(inputs_ids, attention_mask=attention_mask, max_length=512)
    pred = tokenizer.batch_decode(generation, skip_special_tokens=True)
    result = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    current_metrics = compute_metrics_for_test(pred, result)
    test_metrics['rouge1'] += current_metrics['rouge1'] / 30
    test_metrics['rouge2'] += current_metrics['rouge2'] / 30
    test_metrics['rougeL'] += current_metrics['rougeL'] / 30
    test_metrics['rougeLsum'] += current_metrics['rougeLsum'] / 30
    print(f"inputs[#{idx + 1}]: {pred}")
    print(f"label[#{idx + 1}]: {result}")
    print()
    f.write(f"inputs[#{idx + 1}]: {pred}\n")
    f.write(f"label[#{idx + 1}]: {result}\n")
    f.write("-------------------------------------------\n\n")
f.write(f"ROUGE SCORE [ROUGE-1: {test_metrics['rouge1']}, ROUGE-2: {test_metrics['rouge2']}, ROUGE-L: {test_metrics['rougeL']}, ROUGE-LSUM: {test_metrics['rougeLsum']}]\n")
f.close()
    