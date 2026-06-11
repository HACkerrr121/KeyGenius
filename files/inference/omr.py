"""
Optical Music Recognition: sheet-music image -> MusicXML.

Engine priority (based on current open-source landscape):
  1. homr   -- transformer-based, the maintained successor to oemer; most
              robust to photo/scan quality. `pip install homr`.
  2. oemer  -- fallback. Older U-Net + SVM pipeline.
  3. Audiveris -- best for clean 300dpi+ printed scores, but it's a separate
              Java app (needs Java 21+). Run it manually and feed the resulting
              .musicxml straight into predict.py if homr/oemer struggle.

Reality check: ALL open OMR degrades on dense polyphonic piano. For training
you should use PIG's symbolic data directly; OMR is only for the convenience of
accepting a photo at inference time. Prefer MusicXML/MIDI input when you have it.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile


def _find_musicxml(directory: str) -> str | None:
    for f in os.listdir(directory):
        if f.lower().endswith((".musicxml", ".mxl", ".xml")):
            return os.path.join(directory, f)
    return None


def image_to_musicxml(image_path: str, out_dir: str | None = None) -> str:
    """Convert an image to MusicXML. Returns the output path or raises."""
    out_dir = out_dir or tempfile.mkdtemp(prefix="keygenius_omr_")
    os.makedirs(out_dir, exist_ok=True)

    # ---- 1. homr ----------------------------------------------------------
    if shutil.which("homr"):
        try:
            subprocess.run(["homr", image_path], cwd=out_dir,
                           check=True, capture_output=True, text=True)
            found = _find_musicxml(out_dir) or _find_musicxml(os.path.dirname(image_path))
            if found:
                return found
        except subprocess.CalledProcessError as e:
            print(f"[warn] homr failed: {e.stderr[:300]}")

    # ---- 2. oemer (fallback) ---------------------------------------------
    if shutil.which("oemer"):
        try:
            subprocess.run(["oemer", image_path, "-o", out_dir],
                           check=True, capture_output=True, text=True)
            found = _find_musicxml(out_dir)
            if found:
                return found
        except subprocess.CalledProcessError as e:
            print(f"[warn] oemer failed: {e.stderr[:300]}")

    raise RuntimeError(
        "OMR failed. Install homr (`pip install homr`) or oemer, or convert the "
        "image with Audiveris manually and pass the resulting .musicxml to "
        "predict.py directly."
    )
