"""Embed each downloaded cover with CLIP, for the cheap image baseline.

    uv run python -m preflight.embed

Adds new covers to `data/cover_embeddings.npz` (ids and 512-d vectors).
CLIP ViT-B/32 runs through fastembed on ONNX Runtime: CPU only, no torch. It
is a baseline the fine-tune has to beat, so a small, well-known model is the
point rather than a compromise.
"""
import sys

import numpy as np

from preflight.covers import COVERS
from preflight.store import DEFAULT_PATH

EMBEDDINGS = DEFAULT_PATH.parent / "cover_embeddings.npz"
MODEL = "Qdrant/clip-ViT-B-32-vision"
#: The laptop is shared; see `preflight.baseline.THREADS`.
THREADS = 4


def load_embeddings() -> dict[str, np.ndarray]:
    if not EMBEDDINGS.exists():
        return {}
    stored = np.load(EMBEDDINGS)
    return {str(media_id): vector for media_id, vector in zip(stored["ids"], stored["vectors"])}


def main() -> int:
    vectors = load_embeddings()
    new = [path for path in sorted(COVERS.glob("*.jpg")) if path.stem not in vectors]
    if new:
        from fastembed import ImageEmbedding  # slow to import; readers of the file do not need it

        model = ImageEmbedding(MODEL, cache_dir=str(DEFAULT_PATH.parent / "models"), threads=THREADS)
        for path, vector in zip(new, model.embed([str(path) for path in new], batch_size=32)):
            vectors[path.stem] = vector
        ids = sorted(vectors)
        partial = EMBEDDINGS.with_suffix(".part.npz")
        np.savez(partial, ids=np.array(ids), vectors=np.stack([vectors[i] for i in ids]))
        partial.rename(EMBEDDINGS)
    print(f"embedded {len(new)} new covers; {len(vectors)} in {EMBEDDINGS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
