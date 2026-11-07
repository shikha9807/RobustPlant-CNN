"""RobustPlant-CNN: EfficientNet backbone with multi-stage CBAM attention.

Architecture
------------
  EfficientNet (ImageNet-pretrained, include_top=False)
    |-- stage-end feature maps (default: stages 5 and 6) -> CBAM -> GAP
    '-- final feature map (top_activation)               -> CBAM -> GAP
  concat(GAP features) -> Dropout(0.4) -> Dense(logits) -> Softmax

CBAM is attached to the backbone's feature maps instead of being spliced
between pretrained blocks, so the pretrained weights stay untouched and the
ablations are one flag each (``use_cbam=False`` gives the plain backbone,
``cbam_stages=()`` gives final-feature-map attention only).

Layer naming convention (used to separate backbone from new layers):
  ``cbam_*`` and ``head_*`` are new layers; everything else is backbone.
"""
from __future__ import annotations

import inspect
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers

from src import config
from src.cbam import CBAM

BACKBONES = {
    "EfficientNetB0": tf.keras.applications.EfficientNetB0,
    "EfficientNetB3": tf.keras.applications.EfficientNetB3,
}
DEFAULT_CBAM_STAGES = (5, 6)
LOGITS_LAYER = "head_logits"
PROBS_LAYER = "head_softmax"
WEIGHTS_FILE = "best.weights.h5"
CONFIG_FILE = "model_config.json"
_NEW_LAYER_PREFIXES = ("cbam_", "head_")


@dataclass
class ModelConfig:
    backbone: str = config.BACKBONE
    num_classes: int = config.NUM_CLASSES
    use_cbam: bool = True
    cbam_stages: tuple = DEFAULT_CBAM_STAGES
    dropout: float = config.DROPOUT_RATE
    reduction_ratio: int = config.CBAM_REDUCTION_RATIO
    kernel_size: int = config.CBAM_SPATIAL_KERNEL
    input_size: tuple = tuple(config.IMG_SIZE)
    class_names: list = field(default_factory=lambda: list(config.PLANTVILLAGE_CLASSES))

    @property
    def default_run_name(self) -> str:
        tag = "cbam" if self.use_cbam else "base"
        return f"{self.backbone.lower()}_{tag}"


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #
def _stage_end_layers(base: tf.keras.Model) -> dict[int, str]:
    """Map stage number -> name of the layer that ends that stage."""
    pattern = re.compile(r"^block(\d+)([a-z])_(add|project_bn)$")
    best: dict[int, tuple] = {}
    for layer in base.layers:
        match = pattern.match(layer.name)
        if match:
            stage, letter, kind = int(match[1]), match[2], match[3]
            score = (letter, kind == "add")
            if stage not in best or score >= best[stage][0]:
                best[stage] = (score, layer.name)
    return {stage: name for stage, (_, name) in best.items()}


def _make_cbam(name: str, channels: int, reduction_ratio: int, kernel_size: int):
    """Instantiate src.cbam.CBAM, passing only the arguments it accepts."""
    params = inspect.signature(CBAM.__init__).parameters
    takes_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD
                       for p in params.values())
    values = {
        ("reduction_ratio", "reduction", "ratio"): reduction_ratio,
        ("kernel_size", "spatial_kernel_size", "spatial_kernel"): kernel_size,
        ("channels", "filters", "num_channels"): channels,
    }
    kwargs = {}
    for aliases, value in values.items():
        for alias in aliases:
            if alias in params:
                kwargs[alias] = value
                break
    if "name" in params or takes_kwargs:
        kwargs["name"] = name
    return CBAM(**kwargs)


def build_model(cfg: ModelConfig | None = None,
                weights: str | None = "imagenet") -> tf.keras.Model:
    """Build the (optionally attention-augmented) classifier.

    Args:
        cfg: architecture settings.
        weights: "imagenet" for pretrained backbone, None for random init
            (used when restoring a trained checkpoint).
    """
    cfg = cfg or ModelConfig()
    if cfg.backbone not in BACKBONES:
        raise ValueError(f"Unknown backbone {cfg.backbone}; "
                         f"choose from {list(BACKBONES)}")
    input_shape = (*cfg.input_size, config.IMG_CHANNELS)
    base = BACKBONES[cfg.backbone](include_top=False, weights=weights,
                                   input_shape=input_shape)
    stage_layers = _stage_end_layers(base)

    pooled = []
    if cfg.use_cbam:
        missing = [s for s in cfg.cbam_stages if s not in stage_layers]
        if missing:
            raise ValueError(f"Stages {missing} not found; available: "
                             f"{sorted(stage_layers)}")
        for stage in cfg.cbam_stages:
            feat = base.get_layer(stage_layers[stage]).output
            feat = _make_cbam(f"cbam_stage{stage}", int(feat.shape[-1]),
                              cfg.reduction_ratio, cfg.kernel_size)(feat)
            pooled.append(layers.GlobalAveragePooling2D(
                name=f"head_gap_stage{stage}")(feat))

    top = base.get_layer("top_activation").output
    if cfg.use_cbam:
        top = _make_cbam("cbam_top", int(top.shape[-1]), cfg.reduction_ratio,
                         cfg.kernel_size)(top)
    pooled.append(layers.GlobalAveragePooling2D(name="head_gap_top")(top))

    x = pooled[0] if len(pooled) == 1 else layers.Concatenate(
        name="head_concat")(pooled)
    x = layers.Dropout(cfg.dropout, name="head_dropout")(x)
    logits = layers.Dense(cfg.num_classes, name=LOGITS_LAYER)(x)
    probs = layers.Softmax(name=PROBS_LAYER)(logits)
    return tf.keras.Model(base.input, probs, name=f"robustplant_{cfg.default_run_name}")


# --------------------------------------------------------------------------- #
# Freezing / fine-tuning helpers
# --------------------------------------------------------------------------- #
def _is_new_layer(layer: tf.keras.layers.Layer) -> bool:
    return layer.name.startswith(_NEW_LAYER_PREFIXES)


def backbone_layers(model: tf.keras.Model) -> list:
    return [l for l in model.layers if not _is_new_layer(l)]


def set_backbone_trainable(model: tf.keras.Model, trainable: bool) -> None:
    """Freeze/unfreeze the backbone; CBAM and head layers stay trainable."""
    for layer in model.layers:
        layer.trainable = True if _is_new_layer(layer) else trainable


def unfreeze_top_backbone(model: tf.keras.Model, n_layers: int) -> int:
    """Unfreeze the last `n_layers` backbone layers (BatchNorm stays frozen).

    Returns the number of layers actually unfrozen. Call model.compile() again
    afterwards for the change to take effect.
    """
    set_backbone_trainable(model, False)
    count = 0
    for layer in backbone_layers(model)[-n_layers:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
            count += 1
    return count


def count_parameters(model: tf.keras.Model) -> dict:
    trainable = int(sum(tf.keras.backend.count_params(w)
                        for w in model.trainable_weights))
    total = int(model.count_params())
    return {"total": total, "trainable": trainable,
            "non_trainable": total - trainable}


# --------------------------------------------------------------------------- #
# Persistence (weights + JSON config; avoids custom-object deserialisation)
# --------------------------------------------------------------------------- #
def model_dir(run_name: str) -> Path:
    return config.MODELS_DIR / run_name


def save_model_config(run_name: str, cfg: ModelConfig) -> Path:
    path = model_dir(run_name)
    path.mkdir(parents=True, exist_ok=True)
    out = path / CONFIG_FILE
    out.write_text(json.dumps(asdict(cfg), indent=2))
    return out


def load_model_config(run_name: str) -> ModelConfig:
    raw = json.loads((model_dir(run_name) / CONFIG_FILE).read_text())
    raw["cbam_stages"] = tuple(raw["cbam_stages"])
    raw["input_size"] = tuple(raw["input_size"])
    return ModelConfig(**raw)


def load_trained_model(run_name: str,
                       weights_file: str = WEIGHTS_FILE
                       ) -> tuple[tf.keras.Model, ModelConfig]:
    """Rebuild the architecture from JSON and load the best checkpoint."""
    cfg = load_model_config(run_name)
    if cfg.class_names != list(config.PLANTVILLAGE_CLASSES):
        raise ValueError("Checkpoint class order differs from config.py")
    model = build_model(cfg, weights=None)
    model.load_weights(str(model_dir(run_name) / weights_file))
    return model, cfg


def get_logits_model(model: tf.keras.Model) -> tf.keras.Model:
    """Sub-model returning pre-softmax logits (for temperature scaling)."""
    return tf.keras.Model(model.input, model.get_layer(LOGITS_LAYER).output,
                          name=f"{model.name}_logits")


def gradcam_layer_name(model: tf.keras.Model) -> str:
    """Name of the final spatial feature map (target layer for Grad-CAM)."""
    names = {l.name for l in model.layers}
    return "cbam_top" if "cbam_top" in names else "top_activation"
