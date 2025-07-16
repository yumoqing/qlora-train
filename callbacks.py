
from transformers import TrainerCallback, EarlyStoppingCallback
import os

class SaveBestModelCallback(TrainerCallback):
    def __init__(self):
        self.best_metric = None

    def on_evaluate(self, args, state, control, **kwargs):
        metrics = kwargs.get("metrics", {})
        eval_loss = metrics.get("eval_loss")
        if eval_loss is None:
            return
        if self.best_metric is None or eval_loss < self.best_metric:
            print(f"🌟 Best model updated: {self.best_metric} -> {eval_loss}")
            self.best_metric = eval_loss
            best_model_path = os.path.join(args.output_dir, "best_model")
            kwargs["model"].save_pretrained(best_model_path)
            kwargs["tokenizer"].save_pretrained(best_model_path)

def build_callbacks(config):
    return [
        EarlyStoppingCallback(
            early_stopping_patience=config.early_stopping_patience,
            early_stopping_threshold=config.early_stopping_threshold,
        ),
        SaveBestModelCallback(),
    ]
