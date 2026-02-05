from oemer import layers
import os
import json
from collections import defaultdict
import re
from fast_oemer_extract import fast_extract_notes

# Configuration
path_to_data = "/Users/anandkashyap/Documents/GitHub/KeyGenius/Backend/Music_Data/Scores"
out_dir = "/Users/anandkashyap/Documents/GitHub/KeyGenius/Backend/Music_Data/output"
output_json = "/Users/anandkashyap/Documents/GitHub/KeyGenius/Backend/Music_Data/extracted_notes.json"

os.makedirs(out_dir, exist_ok=True)


def group_pages_by_piece(files):
    """
    Group image files by their base piece name.
    Example: '001_Bach_Invention_No1_C_page_0.jpg' and '001_Bach_Invention_No1_C_page_1.jpg'
    will be grouped under '001_Bach_Invention_No1_C'
    """
    pieces = defaultdict(list)

    for file in files:
        if file.startswith('.') or not (file.endswith('.jpg') or file.endswith('.png')):
            continue

        # Extract base name without page number
        # Pattern: anything ending with _page_N.ext
        match = re.match(r'(.+)_page_(\d+)\.(jpg|png)', file)
        if match:
            base_name = match.group(1)
            page_num = int(match.group(2))
            pieces[base_name].append((page_num, file))
        else:
            # Single page file without _page_N suffix
            base_name = os.path.splitext(file)[0]
            pieces[base_name].append((0, file))

    # Sort pages within each piece
    for piece_name in pieces:
        pieces[piece_name].sort(key=lambda x: x[0])

    return pieces


def extract_notes_from_image(img_path):
    """
    Extract notes from a single image using FAST oemer extraction.
    Returns a list of note dictionaries with their properties.
    Skips symbol extraction, rhythm parsing, and MusicXML building for 3-5x speedup.
    """
    try:
        print(f"  Extracting from: {os.path.basename(img_path)}")

        # Use fast extraction that stops as soon as we have note bboxes
        notes = fast_extract_notes(
            img_path=img_path,
            use_tf=True,
            save_cache=False,
            without_deskew=True
        )

        # Extract detailed note information
        extracted_notes = []
        for note in notes:
            if note.bbox:
                note_info = {
                    'bbox': [int(x) for x in note.bbox],  # [x, y, width, height]
                    'pitch': getattr(note, 'pitch', None),
                    'note_type': getattr(note, 'note_type', None),
                    'dots': getattr(note, 'dots', 0),
                }
                extracted_notes.append(note_info)

        print(f"    Found {len(extracted_notes)} notes")
        return extracted_notes

    except Exception as e:
        print(f"    ERROR: {str(e)}")
        return []


def process_all_pieces(max_pieces=None):
    """
    Main function to process all pieces, handling multi-page scores.

    Args:
        max_pieces: Optional limit on number of pieces to process (for testing)
    """
    # Get all files and group by piece
    all_files = [f for f in os.listdir(path_to_data) if not f.startswith('.')]
    pieces = group_pages_by_piece(all_files)

    print(f"Found {len(pieces)} pieces to process")
    if max_pieces:
        print(f"Processing first {max_pieces} pieces for testing")
    print(f"Total image files: {sum(len(pages) for pages in pieces.values())}")
    print("=" * 80)

    all_results = {}

    pieces_to_process = sorted(pieces.items())
    if max_pieces:
        pieces_to_process = pieces_to_process[:max_pieces]

    for i, (piece_name, pages) in enumerate(pieces_to_process, 1):
        print(f"\n[{i}/{len(pieces)}] Processing: {piece_name}")
        print(f"  Pages: {len(pages)}")

        piece_notes = []

        # Process each page of the piece
        for page_num, filename in pages:
            img_path = os.path.join(path_to_data, filename)
            page_notes = extract_notes_from_image(img_path)

            # Add page information to each note
            for note in page_notes:
                note['page'] = page_num
                note['source_file'] = filename

            piece_notes.extend(page_notes)

        all_results[piece_name] = {
            'total_notes': len(piece_notes),
            'num_pages': len(pages),
            'pages': [filename for _, filename in pages],
            'notes': piece_notes
        }

        print(f"  Total notes extracted: {len(piece_notes)}")

    # Save results to JSON
    print("\n" + "=" * 80)
    print(f"Saving results to: {output_json}")
    with open(output_json, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nProcessing complete!")
    print(f"Processed {len(pieces)} pieces")
    print(f"Total notes extracted: {sum(r['total_notes'] for r in all_results.values())}")

    return all_results


if __name__ == "__main__":
    # For testing, process only first 3 pieces
    # Change max_pieces=None to process all pieces
    results = process_all_pieces(max_pieces=3)
