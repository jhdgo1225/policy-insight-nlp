import gc
import os
from pathlib import Path
import json
from tqdm.auto import tqdm
import math
from collections import defaultdict

import numpy as np
import evaluate
from transformers import EarlyStoppingCallback
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from transformers import TrainerCallback
from transformers.trainer_utils import TrainOutput

from datasets import load_dataset
import torch
from torch.utils.data import IterableDataset

class EpochalSeq2SeqTrainer(Seq2SeqTrainer):
    def train(self, resume_from_checkpoint=None, trial=None, **kwargs):
        self.args.disable_tqdm = True

        train_dataloader = self.get_train_dataloader()
        eval_dataloader  = (
            self.get_eval_dataloader()
            if self.args.eval_strategy == "epoch"
            else None
        )
        print(eval_dataloader)
        total_train_batches = len(train_dataloader)
        num_epochs = math.ceil(self.args.num_train_epochs)
        final_metrics = None

        for epoch in range(num_epochs):
            train_bar = tqdm(
                train_dataloader,
                total=total_train_batches,
                desc=f"Epoch {epoch+1}/{num_epochs} ▶︎ Training",
                unit="batch",
                leave=True,
            )
            for step, inputs in enumerate(train_bar):
                self.training_step(model=self.model, inputs=inputs)
                train_bar.update(0)

            train_bar.close()
            
            if eval_dataloader is not None:
                total_eval_batches = len(eval_dataloader)
                eval_bar = tqdm(
                    eval_dataloader,
                    total=total_eval_batches,
                    desc=f"Epoch {epoch+1}/{num_epochs} ▶︎ Evaluating",
                    unit="batch",
                    leave=True,
                )
                metric_sums = defaultdict(float)
                for inputs in eval_bar:
                    _, logits, labels = self.prediction_step(
                        self.model, inputs, prediction_loss_only=False
                    )
                    if isinstance(logits, (tuple, list)):
                        logits = logits[0]
                    if isinstance(labels, (tuple, list)):
                        labels = labels[0]
                    preds = logits.detach().cpu()
                    labels = labels.detach().cpu()
                    batch_metrics = self.compute_metrics((preds, labels))
                    for k, v in batch_metrics.items():
                        metric_sums[k] += v
                    eval_bar.update(0)
                eval_bar.close()

                final_metrics = {
                    k: round(v / total_eval_batches, 3)
                    for k, v in metric_sums.items()
                }
                log_metrics = {
                    f"epoch#{epoch + 1} - {k}": v
                    for k, v in final_metrics.items()
                }
                self.log(log_metrics)

        return TrainOutput(
            global_step = self.state.global_step,
            training_loss = (self.state.log_history or [{}])[-1].get("loss", None),
            metrics=final_metrics
        )


class Vetorizer(IterableDataset):
    def __init__(self, tokenizer, dataset, seq_length, total_count):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.seq_length = seq_length
        self.total_count = total_count
    
    def __iter__(self):
        for data in self.dataset:
            data['text'][0] = "summarize: " + data['text'][0]
            text_concatenated = " ".join(data['text'])
            label = data['label']
            text_tokenized = self.tokenizer(text_concatenated, padding='max_length', max_length=self.seq_length['encoder'], truncation=True)
            label_tokenized = self.tokenizer(label, padding='max_length', max_length=self.seq_length['decoder'], truncation=True)
            data = {
                'input_ids': text_tokenized['input_ids'],
                'attention_mask': text_tokenized['attention_mask'],
                'labels': label_tokenized['input_ids'],
            }
            yield data

    def __len__(self):
        return self.total_count


def create_dataset(tokenizer, domain_data, args, seq_length):
    train_data = load_dataset('json', data_files=domain_data, split='train', streaming=True)
    no_iter_train_data = load_dataset('json', data_files=domain_data, split='train', streaming=False)
    total_train_data_cnt = len(no_iter_train_data)
    del no_iter_train_data
    gc.collect()

    eval_data = load_dataset('json', data_files=domain_data, split='valid', streaming=True)
    no_iter_eval_data = load_dataset('json', data_files=domain_data, split='valid', streaming=False)
    total_eval_data_cnt = len(no_iter_eval_data)
    del no_iter_eval_data
    gc.collect()
    
    train_dataset = Vetorizer(tokenizer, train_data, seq_length, total_train_data_cnt)
    eval_dataset = Vetorizer(tokenizer, eval_data, seq_length, total_eval_data_cnt)
    
    return train_dataset, eval_dataset


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, (tuple, list)):
        preds = preds[0]
    elif isinstance(preds, torch.Tensor):
        preds = preds.argmax(dim=-1)
    predictions = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    references = tokenizer.batch_decode(labels, skip_special_tokens=True)
    rouge = evaluate.load("rouge")
    scores = rouge.compute(
        predictions=predictions,
        references=references,
        rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"],
        use_stemmer=True,
    )
    return scores

model_ckpt = 'gogamza/kobart-base-v2'
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
tokenizer.model_max_length = 2048

domain_name = 'law_small'

domain_data = {
    'train': f'{domain_name}/train_small.jsonl',
    'valid': f'{domain_name}/valid_small.jsonl',
    'test': f'{domain_name}/test_small.jsonl'
}

seq_length = {
    'encoder': 2048,
    'decoder': 512
}

final_args = Seq2SeqTrainingArguments(
    output_dir="./text_summarize_model_arguments",
    warmup_steps=500,
    eval_strategy="epoch",
	save_strategy="epoch",
    logging_steps=100,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=3e-5,
    weight_decay=0.01,
	load_best_model_at_end=True,
	metric_for_best_model="rougeLsum",
	greater_is_better=True
)

train_dataset, eval_dataset = create_dataset(tokenizer, domain_data, final_args, seq_length)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

final_model = AutoModelForSeq2SeqLM.from_pretrained(
        model_ckpt
    ).to(device)

final_trainer = EpochalSeq2SeqTrainer(
    model=final_model,
    args=final_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer,
        model=final_model
	),
    compute_metrics=compute_metrics,
)

print(final_trainer.train())

trainer_ckpt_dir = "./text_summarize_model"
final_trainer.save_model(trainer_ckpt_dir)
final_trainer.save_state()
