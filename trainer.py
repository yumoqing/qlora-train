
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq

class TrainerManager:
    def __init__(self, config, model, tokenizer, train_dataset, eval_dataset):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

    def create_trainer(self):
        args = TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            num_train_epochs=self.config.num_train_epochs,
            learning_rate=self.config.learning_rate,
            lr_scheduler_type=self.config.lr_scheduler_type,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            evaluation_strategy=self.config.evaluation_strategy,
            eval_steps=self.config.eval_steps,
            save_strategy=self.config.save_strategy,
            save_total_limit=self.config.save_total_limit,
            bf16=True,
            report_to="none",
            remove_unused_columns=False,
            deepspeed=self.config.deepspeed,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )

        collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model, padding=True)
        from callbacks import build_callbacks

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=collator,
            callbacks=build_callbacks(self.config),
        )
        return trainer
