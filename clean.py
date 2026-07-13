"""
clean.py
--------
Conservative dataset cleaning: flags (and optionally moves, never deletes)
corrupted, near-duplicate, and heavily blurred images from a cropped
classification dataset laid out as:

    root_dir/
        helmet/*.jpg
        no_helmet/*.jpg

Files are MOVED to review_dir/<same relative path> rather than deleted, so a
human can spot-check what got flagged before anything is permanently removed.
"""

import os
import shutil
from collections import Counter

import cv2
import imagehash
from PIL import Image, UnidentifiedImageError


def _list_images(root_dir):
    for label in sorted(os.listdir(root_dir)):
        label_dir = os.path.join(root_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for fname in sorted(os.listdir(label_dir)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                yield label, fname, os.path.join(label_dir, fname)


def _is_corrupted(path):
    try:
        with Image.open(path) as im:
            im.verify()
        return False
    except (UnidentifiedImageError, OSError):
        return True


def _blur_score(path):
    """Variance of the Laplacian — lower means blurrier."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return cv2.Laplacian(img, cv2.CV_64F).var()


def clean_dataset(root_dir, review_dir, blur_threshold=15.0, hash_size=8, dry_run=True):
    """
    Scan root_dir for corrupted, near-duplicate, and blurry images.

    - Corrupted images: fail PIL's verify() check.
    - Near-duplicates: identical perceptual hash (imagehash.phash) within the
      same class; only the first occurrence is kept, later ones are flagged.
    - Blurry images: Laplacian-variance blur score below blur_threshold.

    If dry_run=False, flagged files are moved (not deleted) into
    review_dir/<label>/<reason>/<filename>, preserving the original.

    Returns a summary dict with per-reason counts.
    """
    seen_hashes = {}  # label -> set of phash strings
    flagged = []  # list of (label, filename, path, reason)

    for label, fname, path in _list_images(root_dir):
        if _is_corrupted(path):
            flagged.append((label, fname, path, "corrupted"))
            continue  # can't hash or blur-score an unreadable file

        try:
            with Image.open(path) as im:
                h = str(imagehash.phash(im, hash_size=hash_size))
        except Exception:
            flagged.append((label, fname, path, "corrupted"))
            continue

        seen = seen_hashes.setdefault(label, set())
        if h in seen:
            flagged.append((label, fname, path, "duplicate"))
            continue
        seen.add(h)

        blur = _blur_score(path)
        if blur < blur_threshold:
            flagged.append((label, fname, path, "blurry"))

    reason_counts = Counter(r for _, _, _, r in flagged)

    if not dry_run:
        for label, fname, path, reason in flagged:
            dest_dir = os.path.join(review_dir, label, reason)
            os.makedirs(dest_dir, exist_ok=True)
            dest_path = os.path.join(dest_dir, fname)
            if os.path.exists(path):
                shutil.move(path, dest_path)

    return {
        "dry_run": dry_run,
        "total_flagged": len(flagged),
        "by_reason": dict(reason_counts),
    }


def class_balance(root_dir):
    """Return {label: image_count} for the current state of root_dir."""
    counts = Counter()
    for label, _, _ in _list_images(root_dir):
        counts[label] += 1
    return dict(counts)
