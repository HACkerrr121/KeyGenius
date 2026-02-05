# KeyGenius - Separated Oemer and Fingering Model Pipeline

## Overview

The system is now cleanly separated into two independent stages:

1. **Note Extraction (Oemer)**: Extracts notes from sheet music images
2. **Fingering Prediction (Model)**: Predicts fingerings for extracted notes

This separation allows you to:
- Process multi-page sheet music correctly
- Run each stage independently
- Reuse extracted notes for multiple model versions
- Debug each component separately

---

## Pipeline Architecture

```
Sheet Music Images (Scores/)
           ↓
    [coordinate_finder.py]
           ↓
   extracted_notes.json  ← Oemer output (stage 1)
           ↓
      [inference.py]
           ↓
 fingering_predictions.json  ← Model predictions (stage 2)
```

---

## Stage 1: Extract Notes with Oemer

### What it does:
- Groups multi-page scores (e.g., `piece_page_0.jpg` + `piece_page_1.jpg`)
- Extracts notes from ALL pages of each piece
- Saves structured JSON with note positions, pitches, and types

### Usage:

```bash
cd Backend
python coordinate_finder.py
```

### Input:
- Images in `Music_Data/Scores/`
- Supports multi-page pieces (e.g., `001_Bach_Invention_No1_C_page_0.jpg`, `001_Bach_Invention_No1_C_page_1.jpg`)
- Single page pieces also supported

### Output:
- `Music_Data/extracted_notes.json`

### Output Format:
```json
{
  "001_Bach_Invention_No1_C": {
    "total_notes": 150,
    "num_pages": 2,
    "pages": ["001_Bach_Invention_No1_C_page_0.jpg", "001_Bach_Invention_No1_C_page_1.jpg"],
    "notes": [
      {
        "bbox": [x, y, width, height],
        "pitch": 60,
        "note_type": "quarter",
        "dots": 0,
        "page": 0,
        "source_file": "001_Bach_Invention_No1_C_page_0.jpg"
      },
      ...
    ]
  },
  ...
}
```

---

## Stage 2: Predict Fingerings

### What it does:
- Loads notes from `extracted_notes.json`
- Converts note data to model input format
- Runs the trained fingering model
- Outputs fingering predictions with confidence scores

### Usage:

```bash
cd Backend
python inference.py
```

### Requirements:
- Stage 1 must be completed first (`extracted_notes.json` must exist)
- Trained model checkpoint must exist at `checkpoints/best_model.pt`

### Input:
- `Music_Data/extracted_notes.json` (from Stage 1)
- `checkpoints/best_model.pt` (trained model)

### Output:
- `Music_Data/fingering_predictions.json`

### Output Format:
```json
{
  "001_Bach_Invention_No1_C": {
    "num_pages": 2,
    "total_notes": 150,
    "predictions": [
      {
        "bbox": [x, y, width, height],
        "pitch": 60,
        "note_type": "quarter",
        "page": 0,
        "predicted_finger": 3,
        "confidence": 0.95
      },
      ...
    ]
  },
  ...
}
```

---

## Configuration

### Coordinate Finder (Oemer)

Edit paths in [coordinate_finder.py](coordinate_finder.py):

```python
path_to_data = "/path/to/Scores"  # Input images
out_dir = "/path/to/output"       # Oemer temp files
output_json = "/path/to/extracted_notes.json"  # Output
```

### Inference (Model)

Edit paths in [inference.py](inference.py):

```python
oemer_json = "/path/to/extracted_notes.json"
model_checkpoint = "/path/to/checkpoints/best_model.pt"
output_json = "/path/to/fingering_predictions.json"
```

---

## Training the Model

To train the fingering prediction model:

```bash
cd Backend
python train.py
```

This uses the pre-annotated fingering files in `Music_Data/FingeringFiles/` for training and creates checkpoints in `checkpoints/`.

---

## Key Features

### Multi-Page Support
- Automatically detects and groups pages (e.g., `piece_page_0.jpg`, `piece_page_1.jpg`)
- Processes all pages of a piece together
- Maintains page information in output for reference

### Complete Separation
- Oemer runs independently of the model
- Can re-run inference without re-extracting notes
- Can update oemer or model without affecting the other

### Error Handling
- Skips files that fail to process
- Continues processing remaining files
- Reports errors clearly in console output

---

## Troubleshooting

### "Oemer output not found"
Run `coordinate_finder.py` first to extract notes from sheet music.

### "Model checkpoint not found"
Train the model using `train.py` first.

### Memory issues with large scores
The current implementation processes pages sequentially, which should handle large multi-page scores. If issues persist, you can process pieces in batches.

### Notes not detected correctly
Oemer may have difficulty with:
- Low quality images
- Handwritten scores
- Non-standard notation

Consider pre-processing images (deskewing, enhancing contrast) before running oemer.

---

## Next Steps

1. **Run Stage 1**: Extract notes from your sheet music
   ```bash
   python coordinate_finder.py
   ```

2. **Train Model** (if not already done): Train on annotated data
   ```bash
   python train.py
   ```

3. **Run Stage 2**: Get fingering predictions
   ```bash
   python inference.py
   ```

4. **Use Results**: The predictions can be used to:
   - Overlay fingerings on sheet music images
   - Generate annotated PDFs
   - Provide real-time fingering suggestions
