
from configs import Config
from model import ModelManager
from dataset import DatasetManager
from trainer import TrainerManager

def main():
    config = Config.from_yaml("config.yaml")

    model_manager = ModelManager(config)
    model, tokenizer = model_manager.load_model_and_tokenizer()

    dataset_manager = DatasetManager(config, tokenizer)
    train_dataset, eval_dataset = dataset_manager.load_data()

    trainer_manager = TrainerManager(config, model, tokenizer, train_dataset, eval_dataset)
    trainer = trainer_manager.create_trainer()

    trainer.train()
    trainer.save_model(config.output_dir)

if __name__ == "__main__":
    main()
