import json
import os

import torch
from safetensors.torch import save_file
from transformers import Trainer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from ..losses import ebd_loss
from ..model import eb_adapter_state_dict, get_eb_adapters, set_eb_context
from ..model.loading import EB_CONFIG_NAME, EB_WEIGHTS_NAME
from .args import parse_layers


def evident_config(args, data_args) -> dict:
    """Everything needed to rebuild the model and its prompts at inference (evident_config.json)."""
    return dict(
        layers=parse_layers(args.adapter_layers),
        bottleneck_dim=args.bottleneck_dim,
        num_slots=args.num_slots,
        num_iterations=args.num_iterations,
        tokens_per_frame=args.tokens_per_frame,
        use_e2v_gate=args.use_e2v_gate,
        timestamp_format=data_args.timestamp_format,
    )


def save_eb_adapter(model, output_dir: str, config: dict, should_save: bool):
    """Write the EB Adapter weights and evident_config.json next to the LoRA weights."""
    state = {}
    for name, param in eb_adapter_state_dict(model).items():
        if hasattr(param, "ds_id"):  # partitioned by DeepSpeed ZeRO-3
            from deepspeed import zero
            with zero.GatheredParameters([param]):
                state[name] = param.detach().cpu().clone()
        else:
            state[name] = param.detach().cpu().clone()

    if not should_save:
        return
    os.makedirs(output_dir, exist_ok=True)
    save_file(state, os.path.join(output_dir, EB_WEIGHTS_NAME))
    with open(os.path.join(output_dir, EB_CONFIG_NAME), "w") as f:
        json.dump(config, f, indent=2)


class EVIDENTTrainer(Trainer):
    """Adds the EBD loss to the grounding cross-entropy: L = L_CE + λ L_EBD."""

    def __init__(self, *args, evident_config: dict, **kwargs):
        super().__init__(*args, **kwargs)
        self.evident_config = evident_config

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        epochs = self.args.lr_schedule_epochs
        if epochs is not None and epochs > self.args.num_train_epochs:
            num_training_steps = int(num_training_steps * epochs / self.args.num_train_epochs)
        return super().create_scheduler(num_training_steps, optimizer=optimizer)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        set_eb_context(
            model,
            subject_token_indices=inputs.pop("subject_token_indices"),
            object_token_indices=inputs.pop("object_token_indices"),
            concept_valid_mask=inputs.pop("concept_valid_mask"),
        )
        # Note: under DeepSpeed bf16 the Trainer has already cast these features to bf16.
        dino_features = inputs.pop("dino_features", None)

        outputs = super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)
        loss = outputs[0] if return_outputs else outputs

        weight = self.args.ebd_loss_weight
        if model.training and weight > 0 and dino_features is not None:
            attn_maps = [a.last_attn for a in get_eb_adapters(model)]
            l_ebd = ebd_loss(attn_maps, dino_features.flatten(0, 1))
            if self.state.global_step % self.args.logging_steps == 0:
                self.log({"ce_loss": loss.detach().item(), "ebd_loss": l_ebd.detach().item()})
            loss = loss + weight * l_ebd

        return (loss, outputs[1]) if return_outputs else loss

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)
        checkpoint_dir = os.path.join(self._get_output_dir(trial=trial), f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}")
        save_eb_adapter(self.model, checkpoint_dir, self.evident_config, self.args.should_save)
