# KeyGenius

Piano fingering prediction. Given the notes of a piece, predict which finger
(1–5, per hand) plays each note. Clean rebuild around a verified dataset, a
reliable OMR engine, and a structural fix that eliminates the pixel-coordinate
extraction that plagued the old pipeline.

## The three decisions that matter

**Dataset — PIG (verified, canonical).**
The [PIG dataset](https://beam.kisarazu.ac.jp/~saito/research/PianoFingeringDataset/)
(Nakamura, Saito, Yoshii, *Information Sciences* 2020) is the standard public
piano-fingering dataset: 150 pieces, 309 fingering files, ~300 notes each,
multiple annotators on the Bach/Mozart/Chopin subsets. Academic / non-profit
use only — cite the paper. Optional scale-up: **ThumbSet** (Zenodo, on request)
for pretraining, then fine-tune on PIG.

**Honest accuracy.** SOTA on PIG is ~65–71% match rate. A clean ~70% here is a
*real* result. The old 99.9% was overfitting from a leaky split — this repo
splits **by piece**, so no piece appears in both train and val.

**OMR — homr first.**
[homr](https://pypi.org/project/homr/) is the maintained, transformer-based
successor to oemer (oemer's own author recommends it) and is far more robust to
image quality. oemer is kept as a fallback; Audiveris is the option for clean
300dpi+ printed scores. **But OMR is inference-only convenience** — training
uses PIG's symbolic data directly. Prefer MusicXML/MIDI input whenever you have it.

**No more coordinate extraction.**
The old overlay step extracted pixel boxes for each notehead and stamped numbers
on the image — which constantly misaligned. Instead, predicted fingerings are
written **back into the MusicXML as real `<fingering>` notation**, and the
engraver (MuseScore) places them correctly by construction.

## Architecture

Encoder-only sequence labeling, one CRF chain per hand:

```
[continuous feats | pitch-class emb | hand emb]
  -> input projection (-> d_model)
  -> sinusoidal positional encoding
  -> LocalContextConv     (multi-kernel convs: scales / arpeggios)
  -> Transformer encoder  (global context across the line)
  -> emission head        (per-note finger scores)
  -> linear-chain CRF     (transition constraints -> coherent path)
```

Loss = `3.0 * CRF_NLL + 1.0 * focal`. The CRF-heavy weighting forces the model
to learn finger **flow**, not just marginal finger frequencies. Each hand is its
own chain (you never transition between hands), and a hand embedding lets the
model learn the left/right asymmetry (thumb is lowest on the right, highest on
the left). ~2.5M params.

## Layout

```
config.py              single source of truth for dims / hyperparams
data/
  pig_parser.py        PIG .txt -> Note objects
  features.py          note sequence -> tensors (shared by train + inference)
  dataset.py           Dataset, collate, piece-level split, transposition aug
model/
  crf.py               masked linear-chain CRF (NLL + Viterbi)
  architecture.py      KeyGenius model + combined loss
train.py               training loop (warmup+cosine, early stop, best ckpt)
evaluate.py            match rate + general match rate + distribution check
inference/
  omr.py               image -> MusicXML (homr -> oemer -> Audiveris note)
  musicxml_io.py       parse + write fingerings back as notation
  predict.py           end-to-end CLI
```

## Run

```bash
pip install -r requirements.txt          # torch, numpy, music21; homr/oemer optional

# 1. download PIG, unzip into data/PIG/
# 2. train
python train.py
#    watch val match climb toward ~0.65–0.72 with a spread finger distribution

# 3. predict (symbolic input — most reliable)
python -m inference.predict song.musicxml -o song_fingered.musicxml
python -m inference.predict song.mid      -o song_fingered.musicxml

# 3b. predict from a photo (runs OMR first)
python -m inference.predict photo.jpg -o out.musicxml --render
```

Open the output `.musicxml` in MuseScore to see engraved fingerings.

## Confidence flagging

`predict` also computes per-note confidence (CRF posterior marginals) and
**colors low-confidence notes red** so you know exactly which spots to check —
the ambiguous chords/runs where the model is basically guessing. It prints a
review report and you can tune the cutoff:

```bash
python -m inference.predict song.musicxml -o out.musicxml --flag-threshold 0.6
```

This is what makes a ~70% model trustworthy in practice: trust the confident
notes, review the flagged ones.

## Reporting the defensible number

```bash
python eval_pig_multiannotator.py --ckpt checkpoints/keygenius_best.pt
```

On PIG's Bach/Mozart/Chopin subsets (4-6 annotators each) this reports:
`M_single` (vs one annotator), `M_gen` (vs any annotator), and the **human
ceiling** (inter-annotator agreement). Report `M_gen` with the ceiling for
context — landing near the ceiling is the honest, strong result on a task where
humans themselves don't fully agree.

## Notes for reporting results

- Report the **piece-level-split val match rate** (and `general_match_rate` for
  multi-annotator pieces). That's the defensible number.
- The transposition augmentation assumes fingering is ~invariant to key, which
  is the standard assumption in this literature but not perfect — ablate it if
  you want to be rigorous.
- Citation to include: Nakamura, Saito, Yoshii, "Statistical Learning and
  Estimation of Piano Fingering," *Information Sciences* 517 (2020): 68–85.
```
