"""
Extract notes from sheet music images using oemer's FULL pipeline.

Flow: image -> oemer (full OMR) -> MusicXML -> music21 parsing -> features

This gives us:
  - Correct pitches (with accidentals + key signatures)
  - Real rhythm/duration values
  - Proper hand assignment (treble vs bass clef / voice)
  - Chord detection from simultaneous note onsets
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import subprocess
import tempfile
import shutil
import numpy as np
from pathlib import Path
from collections import defaultdict

import music21

# Note to MIDI (kept for compatibility)
NOTE_TO_MIDI = {}
for octave in range(0, 9):
    for i, note in enumerate(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']):
        midi_num = octave * 12 + i + 12
        NOTE_TO_MIDI[f"{note}{octave}"] = midi_num
        flat_map = {'C#': 'Db', 'D#': 'Eb', 'F#': 'Gb', 'G#': 'Ab', 'A#': 'Bb'}
        if note in flat_map:
            NOTE_TO_MIDI[f"{flat_map[note]}{octave}"] = midi_num


def run_oemer(img_path, output_dir=None):
    """
    Run oemer's full pipeline on an image.
    Returns path to the generated MusicXML file.
    """
    img_path = str(Path(img_path).resolve())
    
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="oemer_")
    
    # oemer outputs to current directory, so we cd there
    original_dir = os.getcwd()
    os.chdir(output_dir)
    
    try:
        result = subprocess.run(
            ["oemer", img_path],
            capture_output=True,
            text=True,
            timeout=300  # 5 min timeout
        )
        
        if result.returncode != 0:
            print(f"oemer stderr: {result.stderr}")
            raise RuntimeError(f"oemer failed with code {result.returncode}")
        
        # Find the output MusicXML file
        xml_files = list(Path(output_dir).glob("*.musicxml")) + list(Path(output_dir).glob("*.xml"))
        if not xml_files:
            raise FileNotFoundError(f"No MusicXML output found in {output_dir}")
        
        return str(xml_files[0])
    
    finally:
        os.chdir(original_dir)


def parse_musicxml(xml_path):
    """
    Parse MusicXML with music21 to extract note data.
    
    Returns list of dicts matching the training data format:
        - note: str (e.g. 'C#4')
        - midi: int
        - hand: str ('RIGHT' or 'LEFT')
        - start_time: float (in quarter-note beats, converted to seconds)
        - duration: float
        - delta_time: float
        - is_chord: bool
        - chord_size: int
        - chord_position: float
        - x, y, bbox, page: placeholders for compatibility
    """
    score = music21.converter.parse(xml_path)
    
    notes_data = []
    
    # Iterate over parts
    # In piano music: part 0 = right hand (treble), part 1 = left hand (bass)
    parts = list(score.parts)
    
    for part_idx, part in enumerate(parts):
        # Determine hand from clef or part index
        # For piano: first part is typically right hand
        hand = "RIGHT" if part_idx == 0 else "LEFT"
        
        # Check clef to be more robust
        for el in part.recurse():
            if isinstance(el, music21.clef.BassClef):
                hand = "LEFT"
                break
            elif isinstance(el, music21.clef.TrebleClef):
                hand = "RIGHT"
                break
        
        # Extract all notes and chords
        for element in part.recurse().notesAndRests:
            if isinstance(element, music21.note.Rest):
                continue
            
            # Get offset in quarter notes, convert to approximate seconds
            # Using tempo=120 BPM as default (0.5s per beat)
            offset_beats = float(element.offset)
            # Try to get actual tempo
            tempo_mark = element.getContextByClass(music21.tempo.MetronomeMark)
            if tempo_mark:
                seconds_per_beat = 60.0 / tempo_mark.number
            else:
                seconds_per_beat = 0.5  # default 120 BPM
            
            start_time = offset_beats * seconds_per_beat
            dur_seconds = float(element.quarterLength) * seconds_per_beat
            
            if isinstance(element, music21.chord.Chord):
                # It's a chord - extract each note
                chord_notes = list(element.pitches)
                chord_size = len(chord_notes)
                # Sort by MIDI (low to high)
                chord_notes.sort(key=lambda p: p.midi)
                
                for pos, pitch in enumerate(chord_notes):
                    note_name = pitch.nameWithOctave.replace('-', 'b')  # music21 uses '-' for flat
                    notes_data.append({
                        'note': note_name,
                        'midi': pitch.midi,
                        'hand': hand,
                        'start_time': start_time,
                        'duration': dur_seconds,
                        'is_chord': True,
                        'chord_size': chord_size,
                        'chord_position': pos / max(chord_size - 1, 1),
                        'x': int(offset_beats * 50),  # approximate x for compatibility
                        'y': 200 if hand == "RIGHT" else 400,
                        'bbox': [0, 0, 0, 0],
                        'page': 0,
                    })
            
            elif isinstance(element, music21.note.Note):
                pitch = element.pitch
                note_name = pitch.nameWithOctave.replace('-', 'b')
                notes_data.append({
                    'note': note_name,
                    'midi': pitch.midi,
                    'hand': hand,
                    'start_time': start_time,
                    'duration': dur_seconds,
                    'is_chord': False,
                    'chord_size': 1,
                    'chord_position': 0.5,
                    'x': int(offset_beats * 50),
                    'y': 200 if hand == "RIGHT" else 400,
                    'bbox': [0, 0, 0, 0],
                    'page': 0,
                })
    
    # Sort by start_time, then by pitch (for consistent ordering within chords)
    notes_data.sort(key=lambda n: (n['start_time'], n['midi']))
    
    # Detect chords across parts (notes at same time from different parts)
    # Group by start_time with small tolerance
    _detect_cross_part_chords(notes_data, tolerance=0.01)
    
    # Calculate delta_time
    prev_time = -1.0
    for note in notes_data:
        if abs(note['start_time'] - prev_time) > 0.001:
            note['delta_time'] = note['start_time'] - prev_time if prev_time >= 0 else 0.0
            prev_time = note['start_time']
        else:
            note['delta_time'] = 0.0
    
    return notes_data


def _detect_cross_part_chords(notes_data, tolerance=0.01):
    """
    Update chord info for notes that are simultaneous across parts.
    (e.g. right hand and left hand notes at the same time)
    
    Only updates within the same hand - cross-hand simultaneity
    isn't a "chord" for fingering purposes.
    """
    # Group by (approximate time, hand)
    groups = defaultdict(list)
    for note in notes_data:
        time_bucket = round(note['start_time'] / tolerance) * tolerance
        key = (time_bucket, note['hand'])
        groups[key].append(note)
    
    for key, group in groups.items():
        if len(group) > 1:
            # Sort by MIDI within group
            group.sort(key=lambda n: n['midi'])
            for i, note in enumerate(group):
                note['is_chord'] = True
                note['chord_size'] = len(group)
                note['chord_position'] = i / max(len(group) - 1, 1)


def extract_from_image(img_path):
    """
    Extract notes from a single image using oemer's full pipeline.
    
    Returns list of dicts with all features needed for the model.
    """
    print(f"Running oemer full pipeline on {img_path}...")
    
    # Create temp dir for oemer output
    tmp_dir = tempfile.mkdtemp(prefix="oemer_")
    
    try:
        xml_path = run_oemer(img_path, output_dir=tmp_dir)
        print(f"  MusicXML generated: {xml_path}")
        
        notes_data = parse_musicxml(xml_path)
        
        for note in notes_data:
            note['page'] = 0
        
        return notes_data
    
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def extract_from_musicxml(xml_path):
    """
    Extract notes directly from a MusicXML file.
    Useful if you already have MusicXML (e.g. from MuseScore export).
    """
    notes_data = parse_musicxml(xml_path)
    for note in notes_data:
        note['page'] = 0
    return notes_data


def extract_from_pages(img_paths):
    """Extract notes from multiple page images."""
    all_notes = []
    
    for page_num, img_path in enumerate(img_paths):
        print(f"Processing page {page_num + 1}/{len(img_paths)}: {img_path}")
        
        notes_data = extract_from_image(img_path)
        
        # Add page offset for timing continuity
        if all_notes:
            last_time = max(n['start_time'] for n in all_notes)
            page_offset = last_time + 2.0  # 2 second gap between pages
        else:
            page_offset = 0
        
        for note in notes_data:
            note['page'] = page_num
            note['start_time'] += page_offset
        
        all_notes.extend(notes_data)
        print(f"  Found {len(notes_data)} notes")
    
    # Re-sort and recalculate delta_time
    all_notes.sort(key=lambda n: (n['start_time'], n['midi']))
    
    prev_time = -1.0
    for note in all_notes:
        if abs(note['start_time'] - prev_time) > 0.001:
            note['delta_time'] = note['start_time'] - prev_time if prev_time >= 0 else 0.0
            prev_time = note['start_time']
        else:
            note['delta_time'] = 0.0
    
    print(f"Total: {len(all_notes)} notes across {len(img_paths)} pages")
    return all_notes


def extract_from_folder(folder_path, pattern="*.jpg"):
    """Extract notes from all images in a folder."""
    folder = Path(folder_path)
    img_paths = sorted(folder.glob(pattern))
    
    if not img_paths:
        for ext in ['*.png', '*.jpeg', '*.tiff']:
            img_paths = sorted(folder.glob(ext))
            if img_paths:
                break
    
    if not img_paths:
        raise ValueError(f"No images found in {folder_path}")
    
    return extract_from_pages([str(p) for p in img_paths])


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python fast_oemer_extract.py image.jpg          # from image")
        print("  python fast_oemer_extract.py score.musicxml     # from MusicXML directly")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    if input_path.endswith(('.musicxml', '.xml', '.mxl')):
        print("Parsing MusicXML directly...")
        notes_data = extract_from_musicxml(input_path)
    else:
        notes_data = extract_from_image(input_path)
    
    print(f"\nExtracted {len(notes_data)} notes")
    
    # Stats
    right = sum(1 for n in notes_data if n['hand'] == 'RIGHT')
    left = sum(1 for n in notes_data if n['hand'] == 'LEFT')
    chords = sum(1 for n in notes_data if n.get('is_chord', False))
    print(f"Hands: {right} RIGHT, {left} LEFT")
    print(f"Chords: {chords} in chords, {len(notes_data) - chords} single notes")
    
    # Show pitch range
    midis = [n['midi'] for n in notes_data]
    if midis:
        print(f"Pitch range: MIDI {min(midis)}-{max(midis)}")
    
    print("\nFirst 15 notes:")
    for i, n in enumerate(notes_data[:15]):
        print(f"  {n['note']:6s} midi={n['midi']:3d} hand={n['hand']:5s} "
              f"chord={str(n['is_chord']):5s} size={n['chord_size']} pos={n['chord_position']:.2f} "
              f"t={n['start_time']:.3f} dt={n.get('delta_time', 0):.3f} dur={n['duration']:.3f}")