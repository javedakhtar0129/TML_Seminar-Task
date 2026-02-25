# TML Task - Backdoor Detection

**Javed Akhtar**
7062306

## Files

```
Artifacts/
  images/                        # 3,000 training images
  captions.json                  # Captions for all 3,000 images
  unet_deltas.pt                 # UNet weight deltas (backdoored model)
  public_ground_truth.json       # 10 known poisoned image labels
  example_submission.json        # Reference format for submission
  ground_truth.json              # Full ground truth (host only)

Challenger/
  T2IShield_colab.ipynb          # Backdoor detection notebook

Host/
  backdoor_eviledit_colab.ipynb  # Backdoor injection notebook

evaluator.py                     # Evaluates submission against ground truth
Task Description.pdf             # Official task description
```