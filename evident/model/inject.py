"""Insert EB Adapters into the decoder layers of Qwen2.5-VL."""
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn

from .eb_adapter import EBAdapter, EBContext

ADAPTER_KEY = "eb_adapter"


class DecoderLayerWithEBAdapter(nn.Module):
    """Runs `decoder_layer(X) + eb_adapter(X)`.

    This wrapper is deliberately a plain nn.Module: with gradient checkpointing only
    `decoder_layer` is recomputed, so the adapter's attention maps keep their graph for
    the EBD loss. The wrapped layer is not called `base_layer` because PEFT treats
    modules with that attribute as LoRA layers and would unwrap them on merge.
    """

    def __init__(self, decoder_layer: nn.Module, eb_adapter: EBAdapter, image_token_id: int, ctx: EBContext):
        super().__init__()
        self.decoder_layer = decoder_layer
        self.eb_adapter = eb_adapter
        self.image_token_id = image_token_id
        self.ctx = ctx
        # Read by the language model forward to select the attention mask.
        if hasattr(decoder_layer, "attention_type"):
            self.attention_type = decoder_layer.attention_type

    def __getattr__(self, name: str):
        # Forward attribute lookups (e.g. `.mlp`, `.input_layernorm`) to the wrapped
        # layer so that kernel patchers such as Liger still find them.
        try:
            return super().__getattr__(name)
        except AttributeError:
            decoder_layer = self.__dict__.get("_modules", {}).get("decoder_layer")
            if decoder_layer is None:
                raise
            return getattr(decoder_layer, name)

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        outputs = self.decoder_layer(hidden_states, *args, **kwargs)
        base_hidden = outputs[0] if isinstance(outputs, tuple) else outputs

        input_ids = self.ctx.input_ids
        if input_ids is not None and input_ids.shape[1] == hidden_states.shape[1]:
            vision_mask = input_ids == self.image_token_id
            if vision_mask.any():
                residual = self.eb_adapter(hidden_states, vision_mask, self.ctx)
            else:
                residual = self.eb_adapter.forward_text_only(hidden_states)
            base_hidden = base_hidden + residual.to(base_hidden.dtype)

        if isinstance(outputs, tuple):
            return (base_hidden,) + outputs[1:]
        return base_hidden


def get_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Decoder layers of a (possibly PEFT-wrapped) Qwen2.5-VL model."""
    for module in model.modules():
        if type(module).__name__ == "Qwen2_5_VLTextModel":
            return module.layers
    raise ValueError("Could not find the Qwen2.5-VL language model.")


def inject_eb_adapters(
    model: nn.Module,
    layer_indices: Iterable[int],
    bottleneck_dim: int = 896,
    num_slots: int = 4,
    num_iterations: int = 3,
    tokens_per_frame: int = 64,
    use_e2v_gate: bool = True,
) -> EBContext:
    """Wrap the given decoder layers with EB Adapters (in place).

    Returns the EBContext shared by all adapters. The model also keeps a reference
    as `model.eb_context`, and a forward pre-hook keeps `ctx.input_ids` current.
    """
    ctx = EBContext()
    layers = get_decoder_layers(model)
    hidden_dim = model.config.text_config.hidden_size
    image_token_id = model.config.image_token_id

    for i in layer_indices:
        adapter = EBAdapter(
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            num_slots=num_slots,
            num_iterations=num_iterations,
            tokens_per_frame=tokens_per_frame,
            use_e2v_gate=use_e2v_gate,
        )
        layers[i] = DecoderLayerWithEBAdapter(layers[i], adapter, image_token_id, ctx)

    def capture_input_ids(module, args, kwargs):
        # Without input_ids (e.g. inputs_embeds only) the adapters are skipped.
        ctx.input_ids = kwargs.get("input_ids", args[0] if args else None)

    model.register_forward_pre_hook(capture_input_ids, with_kwargs=True)
    model.eb_context = ctx
    return ctx


def get_eb_adapters(model: nn.Module) -> List[EBAdapter]:
    return [m for m in model.modules() if isinstance(m, EBAdapter)]


def set_eb_context(
    model: nn.Module,
    subject_token_indices: Optional[List[List[int]]],
    object_token_indices: Optional[List[List[int]]],
    concept_valid_mask: Optional[torch.Tensor],
):
    ctx = _find_context(model)
    ctx.subject_token_indices = subject_token_indices
    ctx.object_token_indices = object_token_indices
    ctx.concept_valid_mask = concept_valid_mask


def _find_context(model: nn.Module) -> EBContext:
    for module in model.modules():
        if isinstance(module, DecoderLayerWithEBAdapter):
            return module.ctx
    raise ValueError("Model has no EB Adapters.")


def eb_adapter_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """EB Adapter weights keyed relative to the Qwen2.5-VL model, e.g.
    `model.language_model.layers.0.eb_adapter.down_proj.weight`."""
    state = {}
    for name, param in model.named_parameters():
        if f".{ADAPTER_KEY}." in name:
            # Drop PEFT wrapper prefixes ("base_model.model.").
            name = name[name.index("model.language_model."):]
            state[name] = param
    return state


def load_eb_adapter_state_dict(model: nn.Module, state: Dict[str, torch.Tensor]):
    params = {n[n.index("model.language_model."):]: p for n, p in model.named_parameters() if f".{ADAPTER_KEY}." in n}
    missing = sorted(set(params) - set(state))
    unexpected = sorted(set(state) - set(params))
    if missing or unexpected:
        raise KeyError(f"EB Adapter weights do not match the model. missing={missing[:5]} unexpected={unexpected[:5]}")
    with torch.no_grad():
        for name, param in params.items():
            param.copy_(state[name].to(param.dtype))


def lora_target_modules(model: nn.Module, skip_layers: Iterable[int]) -> List[str]:
    """All linear layers of the LLM decoder, except those in `skip_layers` (the EB Adapter layers)."""
    skip = {f"model.language_model.layers.{i}." for i in skip_layers}
    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or not name.startswith("model.language_model.layers."):
            continue
        if any(name.startswith(prefix) for prefix in skip) or f".{ADAPTER_KEY}." in name:
            continue
        targets.append(name)
    return targets
