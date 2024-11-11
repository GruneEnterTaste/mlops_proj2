from datetime import datetime
from typing import Optional

import datasets
import evaluate
import lightning as L
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

import wandb
from pytorch_lightning.loggers import WandbLogger

import os
from sklearn.metrics import f1_score

# Get OS Env Variables
wandb_key = os.getenv("WANDB_API_KEY")
path = os.getenv("SAVE_PATH")

if os.getenv("LR"):
    lr = float(os.getenv("LR"))
else:
    print("No learning Rate was given, using default 8.607559313717729e-05")
    lr = 8.607559313717729e-05

if os.getenv("WARMUP_STEPS"):
    warmup_steps = int(os.getenv("WARMUP_STEPS"))
else:
    print("No Warmup steps were given, using default 50")
    warmup_steps = 50

if os.getenv("DECAY"):
    decay = float(os.getenv("DECAY"))
else:
    print("No weight decay was given, using default 6.202247374050644e-05")
    decay = 6.202247374050644e-05

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class GLUEDataModule(L.LightningDataModule):
    task_text_field_map = {
        "cola": ["sentence"],
        "sst2": ["sentence"],
        "mrpc": ["sentence1", "sentence2"],
        "qqp": ["question1", "question2"],
        "stsb": ["sentence1", "sentence2"],
        "mnli": ["premise", "hypothesis"],
        "qnli": ["question", "sentence"],
        "rte": ["sentence1", "sentence2"],
        "wnli": ["sentence1", "sentence2"],
        "ax": ["premise", "hypothesis"],
    }

    glue_task_num_labels = {
        "cola": 2,
        "sst2": 2,
        "mrpc": 2,
        "qqp": 2,
        "stsb": 1,
        "mnli": 3,
        "qnli": 2,
        "rte": 2,
        "wnli": 2,
        "ax": 3,
    }

    loader_columns = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    ]

    def __init__(
        self,
        model_name_or_path: str,
        task_name: str = "mrpc",
        max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        **kwargs,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.text_fields = self.task_text_field_map[task_name]
        self.num_labels = self.glue_task_num_labels[task_name]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def setup(self, stage: str):
        self.dataset = datasets.load_dataset("glue", self.task_name)

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                remove_columns=["label"],
            )
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)

        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]

    def prepare_data(self):
        datasets.load_dataset("glue", self.task_name)
        AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.train_batch_size, shuffle=True, num_workers=15)

    def val_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["validation"], batch_size=self.eval_batch_size, num_workers=15)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size, num_workers=15) for x in self.eval_splits]

    def test_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size, num_workers=15)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size, num_workers=15) for x in self.eval_splits]

    def convert_to_features(self, example_batch, indices=None):
        # Either encode single sentence or sentence pairs
        if len(self.text_fields) > 1:
            texts_or_text_pairs = list(zip(example_batch[self.text_fields[0]], example_batch[self.text_fields[1]]))
        else:
            texts_or_text_pairs = example_batch[self.text_fields[0]]

        # Tokenize the text/text pairs
        features = self.tokenizer.batch_encode_plus(
            texts_or_text_pairs, max_length=self.max_seq_length, padding="max_length", truncation=True
        )

        # Rename label to labels to make it easier to pass to model forward
        features["labels"] = example_batch["label"]

        return features



class GLUETransformer(L.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        task_name: str,
        learning_rate: float = 2e-5,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        eval_splits: Optional[list] = None,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)
        self.metric = evaluate.load(
            "glue", self.hparams.task_name, experiment_id=datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        )

        self.validation_step_outputs = []

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]

        if self.hparams.num_labels > 1:
            preds = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()

        labels = batch["labels"]
        self.validation_step_outputs.append({"loss": val_loss, "preds": preds, "labels": labels})
        return val_loss

    def on_validation_epoch_end(self):
        preds = torch.cat([x["preds"] for x in self.validation_step_outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in self.validation_step_outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"] for x in self.validation_step_outputs]).mean()

        f1 = f1_score(labels, preds)
        accuracy = (preds == labels).mean()

        self.log('accuracy', accuracy, prog_bar=True)
        self.log('f1', f1, prog_bar=True)
        self.log("val_loss", loss, prog_bar=True)
        # This causes recursion Error somehow:
        # self.log_dict(self.metric.compute(predictions=preds, references=labels), prog_bar=True)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

if wandb_key:
    wandb.login(key=wandb_key)
else:
    print("WANDB_API_KEY environment variable not found. Please provide it to use WandB.")
    print("You can pass it like the following:")
    print("""docker run -e WANDB_API_KEY="your_wandb_api_key" mlops_proj2""")

wandb.init(project="MLOPS Project 2 DWal", name="Run LR: " + str(lr) + " Decay: " + str(decay) + " Warmup: " + str(warmup_steps))

epochs = 3  # do not change this
logger = WandbLogger()

L.seed_everything(0)

dm = GLUEDataModule(
    model_name_or_path="distilbert-base-uncased",
    task_name="mrpc",
)
dm.setup("fit")
model = GLUETransformer(
    model_name_or_path="distilbert-base-uncased",
    num_labels=dm.num_labels,
    eval_splits=dm.eval_splits,
    task_name=dm.task_name,
    learning_rate=lr,
    weight_decay=decay,
    warmup_steps=warmup_steps,
)

trainer = L.Trainer(
    max_epochs=epochs,
    accelerator="auto",
    devices=1,
    logger=logger,
)
trainer.fit(model, datamodule=dm)

# Save the full model:
if path:
    torch.save(model, path)

wandb.finish()
