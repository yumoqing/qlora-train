
from datasets import load_dataset

class DatasetManager:
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

    def preprocess_function(self, examples, max_length):
        return self.tokenizer(examples["input"], max_length=max_length, truncation=True)

    def load_data(self):
        dataset = load_dataset("json", data_files=self.config.dataset_path, split="train")
        dataset = dataset.shuffle(seed=42)
        split_idx = int(len(dataset) * (1.0 - self.config.validation_split_ratio))
        train_dataset = dataset.select(range(split_idx))
        eval_dataset = dataset.select(range(split_idx, len(dataset)))

        train_dataset = train_dataset.map(lambda x: self.preprocess_function(x, self.config.max_seq_length), batched=True)
        eval_dataset = eval_dataset.map(lambda x: self.preprocess_function(x, self.config.max_seq_length), batched=True)
        return train_dataset, eval_dataset
