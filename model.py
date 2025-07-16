
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

class ModelManager:
    def __init__(self, config):
        self.config = config

    def load_model_and_tokenizer(self):
        model_kwargs = {}
        if self.config.quantization_enable:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=self.config.load_in_4bit,
                bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
            )
            model_kwargs["quantization_config"] = bnb_config
            print("✅ 使用量化加载模型")
        else:
            print("🚀 使用全精度加载模型")

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            device_map="auto",
            **model_kwargs
        )
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name_or_path, trust_remote_code=True)

        if self.config.quantization_enable:
            model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=self.config.target_modules,
        )
        model = get_peft_model(model, lora_config)

        return model, tokenizer
