
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import load_from_disk
import argparse
import logging
import random
import torch
import sys
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--train-batch-size', type=int, default=32)
    parser.add_argument('--eval-batch-size', type=int, default=64)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--learning_rate', type=str, default=5e-5)

    # Data, model, and output directories
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--n_gpus', type=str, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--training_dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test_dir', type=str, default=os.environ['SM_CHANNEL_TEST'])

    args, _ = parser.parse_known_args()

    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName('INFO'),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load train and test datasets
    train_dataset = load_from_disk(args.training_dir)
    test_dataset = load_from_disk(args.test_dir)

    logger.info(f'[Loaded train_dataset length is: {len(train_dataset)}]')
    logger.info(f'[Loaded test_dataset length is: {len(test_dataset)}]')
    
    # Compute metrics function for binary classification
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

    # Download model from model hub
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)

    # Define training args
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_steps=args.warmup_steps,
        evaluation_strategy='epoch',
        logging_dir=f'{args.output_data_dir}/logs',
        learning_rate=float(args.learning_rate),
    )

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    # Train model
    trainer.train()

    # Evaluate model
    eval_result = trainer.evaluate(eval_dataset=test_dataset)

    # Write evaluation results to a file which can be accessed later in S3 output
    with open(os.path.join(args.output_data_dir, 'eval_results.txt'), 'w') as writer:
        for key, value in sorted(eval_result.items()):
            writer.write(f'{key} = {value}\n')

    # Save model to S3
    trainer.save_model(args.model_dir)
