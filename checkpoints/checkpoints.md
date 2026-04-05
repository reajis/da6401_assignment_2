# Checkpoint Submission Instructions

Submit the following three checkpoint files inside this folder:

1. Task-1: `classifier.pth`
2. Task-2: `localizer.pth`
3. Task-3: `unet.pth`

These filenames are mandatory for evaluation.

## How These Are Used

- Task-4 (Unified Model): initialize the shared backbone/heads using these trained weights.
- Evaluation pipeline: we will load these same checkpoint files during grading.

## Expected Content

Each `.pth` may store either:

- a plain `state_dict`, or
- a dict containing model weights under the key, `state_dict`.

Recommended checkpoint payload:

```python
{
	"state_dict": model.state_dict(),
	"epoch": epoch,
	"best_metric": best_metric,
}
```

## Notes

- Keep architecture definitions consistent with saved weights.
- Do not rename files.
- Ensure checkpoints are readable with `torch.load(..., map_location=device)`.
