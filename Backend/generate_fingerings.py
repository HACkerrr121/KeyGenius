"""
Generate fingering annotations for MusicXML files using PRamoneda's pretrained models.
Replaces broken fingering files in Music_Data/FingeringFiles/.

Usage:
    python generate_fingerings.py                    # process all score images
    python generate_fingerings.py score.musicxml     # single file
"""
import sys
import os
import subprocess
import tempfile
import shutil
from pathlib import Path

import numpy as np
import torch
import music21

REPO_PATH = Path(__file__).parent.parent.parent / "Automatic-Piano-Fingering"
sys.path.insert(0, str(REPO_PATH))

from nns import seq2seq_model, common
from nns.GGCN import edges_to_matrix


def load_model(path, model, device=None):
    checkpoint = torch.load(path, map_location=device or "cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device or "cpu")
    return model

BACKEND = Path(__file__).parent
SCORES_DIR = BACKEND / "Music_Data" / "Scores"
OUTPUT_DIR = BACKEND / "Music_Data" / "FingeringFiles"
device = "cuda" if torch.cuda.is_available() else "cpu"


def build_gnn_model():
    return seq2seq_model.seq2seq(
        embedding=common.emb_pitch(),
        encoder=seq2seq_model.gnn_encoder(input_size=64),
        decoder=seq2seq_model.AR_decoder(64),
    )


def load_pretrained(hand):
    model = build_gnn_model()
    path = REPO_PATH / "models" / f"{hand}_ArGNN.pth"
    model = load_model(str(path), model, device=device)
    model.eval()
    return model


def next_onset(onset, onsets):
    sorted_onsets = sorted(set(onsets))
    for o in sorted_onsets:
        if o > onset:
            return o
    return None


def compute_edges(onsets, pitches):
    edges = []
    for i, (o_i, p_i) in enumerate(zip(onsets, pitches)):
        next_o = next_onset(o_i, onsets)
        if next_o is not None:
            next_labels = [(i, j, "next") for j, o_j in enumerate(onsets) if o_j == next_o]
            edges.extend(next_labels)
        onset_labels = [(i, j, "onset") for j, o_j in enumerate(onsets) if o_j == o_i and i != j]
        edges.extend(onset_labels)
    return edges


def normalize(arr):
    mn, mx = np.min(arr), np.max(arr)
    if mx == mn:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / (mx - mn)).astype(np.float32)


def predict_hand(model, midis, onsets_sec, durations_sec):
    n = len(midis)
    if n == 0:
        return []

    notes_norm = (np.array(midis, dtype=np.float32) / 127.0).reshape(1, n, 1)
    onsets_norm = normalize(np.array(onsets_sec, dtype=np.float32)).reshape(1, n, 1)
    durations_norm = normalize(np.array(durations_sec, dtype=np.float32)).reshape(1, n, 1)

    edges = compute_edges(list(np.round(np.array(onsets_sec), 4)), midis)
    edge_matrix = edges_to_matrix(edges, n).unsqueeze(0)  # (1, E, n, n)

    notes_t = torch.tensor(notes_norm).float().to(device)
    onsets_t = torch.tensor(onsets_norm).float().to(device)
    durations_t = torch.tensor(durations_norm).float().to(device)
    lengths_t = torch.IntTensor([n]).to(device)
    edge_t = edge_matrix.to(device)

    with torch.no_grad():
        logits = model(notes_t, onsets_t, durations_t, lengths_t, edge_t)

    preds = logits.argmax(dim=2)[0].cpu().tolist()  # 0-4
    return [p + 1 for p in preds]  # 1-5


def musicxml_to_notes(xml_path):
    """Parse MusicXML and return separate right/left hand note lists."""
    score = music21.converter.parse(xml_path)
    parts = list(score.parts)

    right, left = [], []

    for part_idx, part in enumerate(parts):
        hand = "right" if part_idx == 0 else "left"
        for el in part.recurse():
            if isinstance(el, music21.clef.BassClef):
                hand = "left"
            elif isinstance(el, music21.clef.TrebleClef):
                hand = "right"

        tempo_mark = None
        for el in part.recurse():
            if isinstance(el, music21.tempo.MetronomeMark):
                tempo_mark = el
                break
        spb = 60.0 / (tempo_mark.number if tempo_mark else 120.0)

        for el in part.recurse().notesAndRests:
            if isinstance(el, music21.note.Rest):
                continue
            onset_s = float(el.offset) * spb
            dur_s = float(el.quarterLength) * spb

            notes_to_add = []
            if isinstance(el, music21.chord.Chord):
                for p in el.pitches:
                    notes_to_add.append((p.midi, p.nameWithOctave.replace('-', 'b'), onset_s, dur_s))
            elif isinstance(el, music21.note.Note):
                notes_to_add.append((el.pitch.midi, el.pitch.nameWithOctave.replace('-', 'b'), onset_s, dur_s))

            for entry in notes_to_add:
                if hand == "right":
                    right.append(entry)
                else:
                    left.append(entry)

    right.sort(key=lambda x: (x[2], x[0]))
    left.sort(key=lambda x: (x[2], x[0]))
    return right, left


def generate_fingering_file(xml_path, output_path, model_rh, model_lh, piece_id):
    print(f"  Parsing {xml_path}...")
    right, left = musicxml_to_notes(xml_path)
    print(f"  Right: {len(right)} notes, Left: {len(left)} notes")

    rh_fingers = predict_hand(model_rh,
                              [n[0] for n in right],
                              [n[2] for n in right],
                              [n[3] for n in right])

    lh_fingers = predict_hand(model_lh,
                              [n[0] for n in left],
                              [n[2] for n in left],
                              [n[3] for n in left])

    # Merge and sort by onset
    all_notes = []
    for i, (midi, name, onset, dur) in enumerate(right):
        f = rh_fingers[i] if i < len(rh_fingers) else 1
        all_notes.append((onset, dur, name, 64, 80, 0, f))   # hand=0 right

    for i, (midi, name, onset, dur) in enumerate(left):
        f = lh_fingers[i] if i < len(lh_fingers) else 1
        all_notes.append((onset, dur, name, 64, 80, 1, -f))  # hand=1 left, negative finger

    all_notes.sort(key=lambda x: x[0])

    with open(output_path, 'w') as f:
        f.write("//Version: PianoFingering_v170101\n")
        for idx, (onset, dur, name, vel, vel2, hand, finger) in enumerate(all_notes):
            f.write(f"{idx}\t{onset:.6f}\t{onset+dur:.6f}\t{name}\t{vel}\t{vel2}\t{hand}\t{finger}\n")

    print(f"  Wrote {len(all_notes)} notes to {output_path}")


def process_score(img_path, piece_id, model_rh, model_lh):
    from inference import _build_audiveris_args
    from fast_oemer_extract import extract_from_musicxml

    img_path = str(Path(img_path).resolve())
    tmp_dir = tempfile.mkdtemp(prefix="audiveris_fg_")

    try:
        subprocess.run(_build_audiveris_args(tmp_dir, img_path),
                       capture_output=True, text=True, timeout=300)
        stem = Path(img_path).stem
        mxl = Path(tmp_dir) / f"{stem}.mxl"
        if not mxl.exists():
            print(f"  WARNING: Audiveris produced no MusicXML for {img_path}")
            return

        out_path = OUTPUT_DIR / f"{piece_id}-generated_fingering.txt"
        generate_fingering_file(str(mxl), str(out_path), model_rh, model_lh, piece_id)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    print("Loading pretrained models...")
    model_rh = load_pretrained("right")
    model_lh = load_pretrained("left")
    print("Models loaded.")

    if len(sys.argv) > 1:
        # Single MusicXML file
        xml_path = sys.argv[1]
        out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/generated_fingering.txt"
        generate_fingering_file(xml_path, out, model_rh, model_lh, "test")
        print(f"Done. Output: {out}")
    else:
        # Process all score images
        images = sorted(SCORES_DIR.glob("*.jpg"))
        print(f"Found {len(images)} score images")

        for img in images:
            # piece id = first 3 digits of filename
            piece_id = img.stem[:3]
            print(f"\nProcessing {img.name} (piece {piece_id})...")
            process_score(str(img), piece_id, model_rh, model_lh)

        print("\nDone. Generated fingering files are in Music_Data/FingeringFiles/")
