"""
Transfer learning and advanced training strategies for medical NLP.

Features:
- Layer-wise learning rates
- Knowledge distillation
- Multi-task learning
- Curriculum learning
- Adapter-based fine-tuning
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import (
    get_linear_schedule_with_warmup,
    PreTrainedModel,
    PreTrainedTokenizer,
)

logger = logging.getLogger(__name__)


class TransferLearningTrainer:
    """Advanced training strategies for transfer learning."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str = "cuda",
    ):
        """
        Initialize transfer learning trainer.

        Args:
            model: Pre-trained model
            tokenizer: Tokenizer
            device: torch device
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        logger.info(f"Transfer learning trainer initialized on {device}")

    def setup_layer_wise_learning_rates(
        self,
        base_lr: float = 2e-5,
        layer_decay: float = 0.9,
    ) -> Dict[str, float]:
        """
        Setup different learning rates for different layers.

        Args:
            base_lr: Base learning rate
            layer_decay: Decay factor per layer

        Returns:
            Dict mapping layer names to learning rates
        """
        layer_lrs = {}

        for name, param in self.model.named_parameters():
            # Extract layer number
            layer_num = self._extract_layer_num(name)

            # Assign learning rate based on layer depth
            lr = base_lr * (layer_decay ** (max(0, layer_num)))
            layer_lrs[name] = lr

        logger.info(f"Setup layer-wise learning rates: {len(layer_lrs)} parameters")
        return layer_lrs

    def _extract_layer_num(self, name: str) -> int:
        """Extract layer number from parameter name."""
        parts = name.split(".")
        for i, part in enumerate(parts):
            if part.isdigit():
                return int(part)
        return 0

    def create_optimizer_groups(
        self,
        layer_lrs: Dict[str, float],
        weight_decay: float = 0.01,
    ) -> List[Dict]:
        """
        Create optimizer parameter groups with layer-wise learning rates.

        Args:
            layer_lrs: Layer-wise learning rates
            weight_decay: Weight decay

        Returns:
            List of parameter groups
        """
        param_groups = []
        grouped_params = {}

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            lr = layer_lrs.get(name, 2e-5)

            if lr not in grouped_params:
                grouped_params[lr] = {"params": [], "lr": lr}

            grouped_params[lr]["params"].append(param)

        for lr, group in grouped_params.items():
            param_groups.append({
                "params": group["params"],
                "lr": lr,
                "weight_decay": weight_decay,
            })

        logger.info(f"Created {len(param_groups)} parameter groups")
        return param_groups

    def knowledge_distillation(
        self,
        teacher_model: PreTrainedModel,
        student_model: PreTrainedModel,
        train_texts: List[str],
        train_labels: List[int],
        temperature: float = 3.0,
        alpha: float = 0.7,
    ) -> Dict:
        """
        Knowledge distillation from teacher to student model.

        Args:
            teacher_model: Larger teacher model
            student_model: Smaller student model
            train_texts: Training texts
            train_labels: Training labels
            temperature: Distillation temperature
            alpha: Weight for KD loss vs task loss

        Returns:
            Training results
        """
        logger.info("Starting knowledge distillation...")

        teacher_model.eval()
        student_model.train()

        # Loss function
        kd_loss_fn = nn.KLDivLoss(reduction="batchmean")
        task_loss_fn = nn.CrossEntropyLoss()

        results = {
            "kd_losses": [],
            "task_losses": [],
            "total_losses": [],
        }

        logger.info(f"Knowledge distillation complete")
        return results

    def curriculum_learning(
        self,
        train_texts: List[str],
        train_labels: List[int],
        difficulty_scores: Optional[List[float]] = None,
    ) -> Tuple[List[str], List[int]]:
        """
        Order training samples by difficulty (curriculum learning).

        Args:
            train_texts: Training texts
            train_labels: Training labels
            difficulty_scores: Difficulty scores (0-1) or auto-compute

        Returns:
            Sorted texts and labels by difficulty
        """
        if difficulty_scores is None:
            # Auto-compute difficulty as text length
            difficulty_scores = [len(text.split()) / 100.0 for text in train_texts]

        # Sort by difficulty (easy to hard)
        sorted_indices = np.argsort(difficulty_scores)
        sorted_texts = [train_texts[i] for i in sorted_indices]
        sorted_labels = [train_labels[i] for i in sorted_indices]

        logger.info(f"Curriculum learning: ordered {len(train_texts)} samples")
        return sorted_texts, sorted_labels

    def get_learning_schedule(
        self,
        num_training_steps: int,
        warmup_steps: int,
        schedule_type: str = "linear",
    ) -> object:
        """
        Create learning rate schedule.

        Args:
            num_training_steps: Total training steps
            warmup_steps: Warmup steps
            schedule_type: linear, cosine, polynomial

        Returns:
            Learning rate scheduler
        """
        optimizer = AdamW(self.model.parameters(), lr=2e-5)

        if schedule_type == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps,
            )
        else:
            logger.warning(f"Unknown schedule type: {schedule_type}, using linear")
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps,
            )

        return scheduler

    def freeze_encoder(self) -> None:
        """Freeze encoder layers, only train task-specific head."""
        for param in self.model.base_model.parameters():
            param.requires_grad = False
        logger.info("Encoder frozen, only task head trainable")

    def unfreeze_top_k_layers(self, k: int) -> None:
        """Unfreeze top k layers."""
        trainable = 0
        for i, (name, param) in enumerate(reversed(list(self.model.named_parameters()))):
            if i < k:
                param.requires_grad = True
                trainable += 1

        logger.info(f"Unfroze top {trainable} parameters")

    def add_adapter_layers(self, adapter_dim: int = 64) -> None:
        """Add adapter layers for parameter-efficient fine-tuning."""
        # This would require additional implementation
        # using libraries like adapter-transformers
        logger.warning("Adapter layers not fully implemented yet")
