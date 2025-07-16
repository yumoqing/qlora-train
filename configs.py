
import yaml
from dataclasses import dataclass

@dataclass
class Config:
    model_name_or_path: str
    output_dir: str
    dataset_path: str
    validation_split_ratio: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    num_train_epochs: int
    learning_rate: float
    lr_scheduler_type: str
    warmup_steps: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: list
    quantization_enable: bool
    load_in_4bit: bool
    bnb_4bit_compute_dtype: str
    bnb_4bit_quant_type: str
    bnb_4bit_use_double_quant: bool
    logging_steps: int
    save_steps: int
    evaluation_strategy: str
    eval_steps: int
    save_strategy: str
    save_total_limit: int
    max_seq_length: int
    early_stopping_patience: int
    early_stopping_threshold: float
    deepspeed: str

    @classmethod
    def from_yaml(cls, file_path):
        with open(file_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
