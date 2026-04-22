from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

from llm_pruning_mmlu.finetuning.config import FinetuningConfig
from llm_pruning_mmlu.finetuning.mask_policy import MaskEnforcer, MaskEnforcerCallback
from llm_pruning_mmlu.finetuning.wandb_utils import WandbRun, report_to_flag


def _data_collator(tokenizer) -> DataCollatorForSeq2Seq:
    """Pad input_ids, attention_mask, and labels to the longest sequence in the batch.

    DataCollatorForSeq2Seq pads labels with -100 so the model only computes
    loss on the answer tokens we labelled in MmluSftDataset.
    """
    return DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
    )


def train_with_masks(
    model: torch.nn.Module,
    tokenizer,
    train_dataset,
    val_dataset,
    ft_cfg: FinetuningConfig,
    enforcer: MaskEnforcer,
    output_dir: Path,
    run_name: str,
    wandb_run: WandbRun,
) -> dict[str, Any]:
    """Run the masked SFT training loop and return training stats.

    The MaskEnforcerCallback re-applies pruning masks after every optimizer
    step.  This is the only structural difference from a standard Trainer run —
    everything else (data collation, LoRA, wandb) is wired through standard HF
    Trainer interfaces, so swapping structured → semi-structured only requires
    passing a different MaskEnforcer.
    """
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=ft_cfg.epochs,
        per_device_train_batch_size=ft_cfg.batch_size,
        per_device_eval_batch_size=ft_cfg.batch_size * 2,
        gradient_accumulation_steps=ft_cfg.gradient_accumulation_steps,
        learning_rate=ft_cfg.learning_rate,
        warmup_ratio=ft_cfg.warmup_ratio,
        weight_decay=ft_cfg.weight_decay,
        bf16=ft_cfg.bf16,
        gradient_checkpointing=ft_cfg.gradient_checkpointing,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=False,
        save_total_limit=1,
        logging_steps=10,
        report_to=report_to_flag(ft_cfg.wandb),
        run_name=run_name,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    callbacks = []
    if enforcer.active:
        callbacks.append(MaskEnforcerCallback(enforcer))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=_data_collator(tokenizer),
        callbacks=callbacks,
    )

    train_result = trainer.train()
    eval_result = trainer.evaluate()

    stats: dict[str, Any] = {
        "train_loss": train_result.training_loss,
        "train_runtime_s": train_result.metrics.get("train_runtime"),
        "train_samples_per_second": train_result.metrics.get("train_samples_per_second"),
        "eval_loss": eval_result.get("eval_loss"),
        "global_step": train_result.global_step,
        "mask_enforcer_active": enforcer.active,
        "pruning_kind": enforcer.pruning_kind,
    }

    wandb_run.log_summary(
        {k: v for k, v in stats.items() if isinstance(v, (int, float, bool))}
    )
    return stats
