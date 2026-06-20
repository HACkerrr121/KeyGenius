"""
Central configuration for KeyGenius.

Everything that more than one module needs to agree on lives here so the
data pipeline, model, and inference code can never silently drift apart
(the #1 cause of the old pipeline's bugs).
"""
from dataclasses import dataclass, field


# ----- task constants ---------------------------------------------------------
NUM_FINGERS = 5            # fingers 1..5 -> classes 0..4
N_PITCH_CLASSES = 12       # C..B
N_HANDS = 2               # 0 = right, 1 = left
MIDI_MIN, MIDI_MAX = 21, 108   # A0..C8, the 88-key range


@dataclass
class FeatureConfig:
    # number of continuous (float) features per note. MUST match features.py.
    n_continuous: int = 14
    pitch_class_emb: int = 8
    hand_emb: int = 4


@dataclass
class ModelConfig:
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    ffn_dim: int = 512
    dropout: float = 0.2
    conv_kernels: tuple = (3, 5, 7)   # LocalContextConv multi-scale kernels
    # CRF transitions are initialized in this range; wider => learns flow faster.
    crf_init_range: float = 0.5


@dataclass
class TrainConfig:
    epochs: int = 80
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    max_seq_len: int = 512          # notes per hand-sequence; longer get windowed
    val_fraction: float = 0.15
    seed: int = 1337
    # loss = crf_weight * crf_nll + focal_weight * focal
    # CRF-heavy weighting forces the model to learn finger TRANSITIONS, not just
    # marginal finger frequencies (the fix that unstuck the old model).
    crf_weight: float = 3.0
    focal_weight: float = 1.0
    focal_gamma: float = 2.0
    grad_clip: float = 1.0
    # pitch transposition augmentation range (semitones), inclusive
    transpose_range: tuple = (-6, 6)
    early_stop_patience: int = 12


@dataclass
class Paths:
    pig_root: str = "data/PIG"          # where you unzip the PIG dataset
    fingering_glob: str = "**/*_fingering*.txt"
    checkpoint_dir: str = "checkpoints"
    best_ckpt: str = "checkpoints/keygenius_best.pt"


@dataclass
class Config:
    feat: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    paths: Paths = field(default_factory=Paths)

    @property
    def input_dim(self) -> int:
        """Total per-note input width fed into the model's input projection."""
        return (self.feat.n_continuous
                + self.feat.pitch_class_emb
                + self.feat.hand_emb)


CFG = Config()