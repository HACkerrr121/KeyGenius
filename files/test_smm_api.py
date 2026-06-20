"""
Quick test of the Sheet Music Manager OMR API.

Goal: find out in one run whether this API is (a) alive and (b) any good,
before we consider building on it. If it's bad or dead, fall back to ACE
(by hand) -> MusicXML -> predict.py.

Usage:
    export SMM_API_KEY=your_key_here          # get one at sheetmusicmanager.com
    python test_smm_api.py /path/to/score.jpg

It will POST the file, save the returned MusicXML next to it, and print a
short verdict. Then run your model on the saved .musicxml to judge accuracy:
    python -m inference.predict score.smm.musicxml -o fingered.musicxml
"""
from __future__ import annotations

import os
import sys
import time

import requests   # pip install requests

API_URL = "https://api.sheetmusicmanager.com/v1/upload"


def main():
    if len(sys.argv) < 2:
        print("usage: python test_smm_api.py <image_or_pdf>")
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"file not found: {path}")
        sys.exit(1)

    api_key = os.environ.get("SMM_API_KEY")
    if not api_key:
        print("Set your key first:  export SMM_API_KEY=your_key_here")
        print("(free tier = 10 conversions/month at sheetmusicmanager.com)")
        sys.exit(1)

    print(f"uploading {os.path.basename(path)} ...")
    t0 = time.time()
    try:
        with open(path, "rb") as f:
            resp = requests.post(
                API_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                files={"file": f},
                data={"format": "both"},     # ask for MusicXML + MIDI
                timeout=180,
            )
    except requests.exceptions.RequestException as e:
        print(f"\nVERDICT: API unreachable / likely dead -> use ACE by hand.\n  ({e})")
        sys.exit(1)

    dt = time.time() - t0
    print(f"HTTP {resp.status_code} in {dt:.1f}s")

    if resp.status_code != 200:
        print(f"\nVERDICT: API errored (status {resp.status_code}) -> use ACE by hand.")
        print(resp.text[:500])
        sys.exit(1)

    try:
        data = resp.json()
    except ValueError:
        print("\nVERDICT: non-JSON response -> treat as dead, use ACE by hand.")
        print(resp.text[:500])
        sys.exit(1)

    musicxml = (data.get("data") or {}).get("musicXML")
    if not musicxml:
        print("\nVERDICT: no MusicXML in response -> use ACE by hand.")
        print(str(data)[:500])
        sys.exit(1)

    out = os.path.splitext(path)[0] + ".smm.musicxml"
    with open(out, "w", encoding="utf-8") as f:
        f.write(musicxml)

    # crude sanity signal: how many notes did it find?
    note_count = musicxml.count("<note")
    print(f"\nSaved -> {out}")
    print(f"rough note count in output: {note_count}")
    print("\nVERDICT: API is ALIVE. Now judge accuracy:")
    print(f"  python -m inference.predict {out} -o fingered.musicxml")
    print("Open fingered.musicxml in MuseScore and compare to what ACE gave you.")
    print("If the notes are clean -> build on this API. If messy -> use ACE by hand.")


if __name__ == "__main__":
    main()