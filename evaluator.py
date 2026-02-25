"""
evaluator.py — BackdoorHunter Challenge (TML Seminar 2025/26)

Evaluates challenger submissions for the backdoor detection task.
Called by the API on every submission.

Expected submission format (JSON):
{
    "team_name": "YourTeamName",
    "ranked_predictions": [
        {"image": "filename1.jpg", "confidence": 0.95},
        {"image": "filename2.jpg", "confidence": 0.12},
        ...   # ALL 3000 images required
    ]
}

Returns:
    dict  — {"score": <public_AP>, "score_held_out": <private_AP>}
    str   — error message if validation fails
"""

import os
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Ground truth must sit in the same directory as this file on the API server.
GROUND_TRUTH_PATH = Path(__file__).parent / "ground_truth.json"

MAX_UPLOAD_BYTES = 10 * 1024 * 1024   # 10 MB  (3000 entries ≈ ~900 KB in practice)
TOTAL_IMAGES     = 3000
NUM_POISONED     = 30
NUM_PUBLIC       = 10
NUM_PRIVATE      = 20


# ---------------------------------------------------------------------------
# Sanity-check helpers (file level)
# ---------------------------------------------------------------------------

def _ext_is_json(path: str) -> bool:
    """Return True if the file has a .json extension."""
    return os.path.splitext(path)[1].lower() == ".json"


def _check_size(path: str, max_bytes: int) -> "str | None":
    """Return an error string if the file exceeds max_bytes, else None."""
    try:
        size = os.path.getsize(path)
        if size > max_bytes:
            return (
                f"File too large: {size:,} bytes "
                f"(limit is {max_bytes:,} bytes / {max_bytes // 1024 // 1024} MB)."
            )
    except Exception as e:
        return f"Could not read uploaded file metadata: {e!r}"
    return None


# ---------------------------------------------------------------------------
# Ground-truth loader (LRU-cached — only read from disk once per process)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_gt() -> dict:
    """
    Load and validate the ground truth JSON.
    Cached after the first call so the file is not re-read on every submission.
    Raises RuntimeError with a clear message on any problem.
    """
    if not GROUND_TRUTH_PATH.exists():
        raise RuntimeError(
            f"Ground truth file not found at '{GROUND_TRUTH_PATH}'. "
            "Ensure it is deployed alongside evaluator.py."
        )

    try:
        with open(GROUND_TRUTH_PATH, "r") as f:
            gt = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Could not parse ground truth JSON: {e!r}")

    # Validate required fields
    required = [
        "public_poisoned", "private_poisoned", "all_poisoned_images",
        "num_total_images", "evaluation_info",
    ]
    for field in required:
        if field not in gt:
            raise RuntimeError(f"Internal GT error: missing required field '{field}'.")

    # Validate counts
    if len(gt["all_poisoned_images"]) != NUM_POISONED:
        raise RuntimeError(
            f"Internal GT error: expected {NUM_POISONED} poisoned images, "
            f"got {len(gt['all_poisoned_images'])}."
        )
    if len(gt["public_poisoned"]) != NUM_PUBLIC:
        raise RuntimeError(
            f"Internal GT error: expected {NUM_PUBLIC} public images, "
            f"got {len(gt['public_poisoned'])}."
        )
    if len(gt["private_poisoned"]) != NUM_PRIVATE:
        raise RuntimeError(
            f"Internal GT error: expected {NUM_PRIVATE} private images, "
            f"got {len(gt['private_poisoned'])}."
        )
    if gt["num_total_images"] != TOTAL_IMAGES:
        raise RuntimeError(
            f"Internal GT error: num_total_images={gt['num_total_images']}, "
            f"expected {TOTAL_IMAGES}."
        )

    # Validate evaluation_info weights
    ei = gt["evaluation_info"]
    for key in ("public_weight", "private_weight"):
        if key not in ei:
            raise RuntimeError(f"Internal GT error: evaluation_info missing '{key}'.")

    return gt


# ---------------------------------------------------------------------------
# Submission-content validator
# ---------------------------------------------------------------------------

def run_assertions(predictions: list, gt: dict) -> "str | None":
    """
    Validate the contents of ranked_predictions.
    Returns an error string on the first problem found, or None if all checks pass.
    """
    try:
        # 1. Correct total count
        if len(predictions) != gt["num_total_images"]:
            return (
                f"'ranked_predictions' must contain exactly {gt['num_total_images']} entries "
                f"(one per image in the dataset), got {len(predictions)}."
            )

        # 2. Each entry must be a dict with 'image' and 'confidence'
        images = []
        for i, pred in enumerate(predictions):
            if not isinstance(pred, dict):
                return (
                    f"Entry at index {i} must be a JSON object, "
                    f"got {type(pred).__name__!r}."
                )
            if "image" not in pred:
                return f"Entry at index {i} is missing the 'image' key."
            if "confidence" not in pred:
                return f"Entry at index {i} is missing the 'confidence' key."

            img  = pred["image"]
            conf = pred["confidence"]

            # image must be a non-empty string
            if not isinstance(img, str) or not img.strip():
                return f"'image' at index {i} must be a non-empty string, got {img!r}."

            # confidence must be a finite float in [0, 1]
            if not isinstance(conf, (int, float)):
                return (
                    f"'confidence' at index {i} must be a number, "
                    f"got {type(conf).__name__!r}."
                )
            if math.isnan(conf) or math.isinf(conf):
                return f"'confidence' at index {i} must be finite, got {conf!r}."
            if not (0.0 <= conf <= 1.0):
                return (
                    f"'confidence' at index {i} is out of range [0, 1]: got {conf}. "
                    "All confidence scores must be between 0 and 1."
                )

            images.append(img)

        # 3. No duplicate image names
        if len(set(images)) != len(images):
            seen = set()
            dupes = [img for img in images if img in seen or seen.add(img)]
            return (
                f"Duplicate image names found in 'ranked_predictions'. "
                f"Examples: {sorted(set(dupes))[:5]}."
            )

        # 4. All 30 known poisoned images must be present.
        #    Acts as a proxy check that correct dataset filenames were used.
        submitted_set = set(images)
        all_poisoned  = set(gt["all_poisoned_images"])
        missing       = all_poisoned - submitted_set
        if missing:
            return (
                f"{len(missing)} known dataset filename(s) are absent from your submission. "
                f"Examples: {sorted(missing)[:5]}. "
                "Ensure you include all 3,000 dataset images using the exact filenames."
            )

    except Exception as e:
        return f"Submission validation failed unexpectedly: {e!r}"

    return None


# ---------------------------------------------------------------------------
# Metric computation 
# ---------------------------------------------------------------------------

def _compute_ap(predictions: list, poisoned_set: set) -> float:
    """
    Compute Average Precision (AP) of the submission against a set of poisoned images.
    Pure numpy implementation — no additional dependencies required.

    Sorts predictions by confidence descending, then computes:
        AP = (1 / |relevant|) * sum_k [ P@k * is_relevant(k) ]
    """
    images      = [p["image"]                      for p in predictions]
    confidences = np.array([float(p["confidence"]) for p in predictions])
    y_true      = np.array([1 if img in poisoned_set else 0 for img in images])

    num_relevant = int(y_true.sum())
    if num_relevant == 0:
        return 0.0

    # Sort by confidence descending
    order    = np.argsort(-confidences)
    y_sorted = y_true[order]

    # Precision at each rank where a relevant item appears
    cumulative_tp  = np.cumsum(y_sorted)
    ranks          = np.arange(1, len(y_sorted) + 1)
    precision_at_k = cumulative_tp / ranks

    ap = float((precision_at_k * y_sorted).sum() / num_relevant)
    return ap


def _compute_scores(predictions: list, gt: dict) -> dict:
    """
    Compute public AP (score) and private AP (score_held_out).

    score           → AP on the 10 public poisoned images  (shown on leaderboard during challenge)
    score_held_out  → AP on the 20 private poisoned images (revealed at final evaluation)

    Final score formula (for reference):
        Final = 0.33 × score + 0.67 × score_held_out
    """
    public_poisoned  = set(gt["public_poisoned"])
    private_poisoned = set(gt["private_poisoned"])

    score          = _compute_ap(predictions, public_poisoned)
    score_held_out = _compute_ap(predictions, private_poisoned)

    return {
        "score":          round(score,          6),
        "score_held_out": round(score_held_out,  6),
    }


# ---------------------------------------------------------------------------
# Main evaluator entry point
# ---------------------------------------------------------------------------

def evaluator(payload: dict) -> "dict | str":
    """
    Main evaluation function called by the API on every submission.

    Args:
        payload: dict with key "file_path" pointing to the uploaded submission file.

    Returns:
        dict  {"score": <float>, "score_held_out": <float>}  on success.
        str   error message on any validation or evaluation failure.
    """
    path = payload["file_path"]

    # ── File-level checks ──────────────────────────────────────────────────
    if not _ext_is_json(path):
        return "Incorrect file type: submission must be a .json file."

    size_err = _check_size(path, MAX_UPLOAD_BYTES)
    if size_err:
        return size_err

    # ── Load ground truth (cached after first call) ─────────────────────────
    try:
        gt = _get_gt()
    except Exception as e:
        return f"Internal ground truth error: {e!r}"

    # ── Load submission JSON ────────────────────────────────────────────────
    try:
        with open(path, "r", encoding="utf-8") as f:
            submission = json.load(f)
    except Exception as e:
        return f"Failed to parse submission JSON: {e!r}"

    # ── Top-level structure checks ──────────────────────────────────────────
    if not isinstance(submission, dict):
        return "Submission must be a JSON object at the top level."

    if "team_name" not in submission:
        return "Submission is missing the required 'team_name' field."

    if "ranked_predictions" not in submission:
        return (
            "Submission is missing the required 'ranked_predictions' field. "
            "See example_submission.json for the expected format."
        )

    if not isinstance(submission["ranked_predictions"], list):
        return "'ranked_predictions' must be a JSON array."

    predictions = submission["ranked_predictions"]

    # ── Content-level validation ────────────────────────────────────────────
    err = run_assertions(predictions, gt)
    if err:
        return err

    # ── Compute and return scores ───────────────────────────────────────────
    return _compute_scores(predictions, gt)


# ---------------------------------------------------------------------------
# Local test  (run: python3 evaluator.py <path_to_submission.json>)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python3 evaluator.py <path_to_submission.json>")
        sys.exit(1)

    result = evaluator({"file_path": sys.argv[1]})
    print("\n── Evaluation Result ──────────────────────────────────")
    print(json.dumps(result, indent=2) if isinstance(result, dict) else f"ERROR: {result}")

    if isinstance(result, dict):
        ei = _get_gt()["evaluation_info"]
        final = (
            ei["public_weight"] * result["score"]
            + ei["private_weight"] * result["score_held_out"]
        )
        print(f"\nFinal Score  (0.33 × public + 0.67 × private): {final:.6f}")
