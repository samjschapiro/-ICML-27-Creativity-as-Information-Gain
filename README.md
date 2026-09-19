# Creativity as Information Gain (ICML 27)

Theory paper with empirical estimators: combinatorial creativity (association, analogy, blending)
read through algorithmic information theory, linking KOMBINE's creativity scores to Elmoznino et
al.'s complexity-based compositionality. The `analogy_ig` track measures how many bits an analogy
carries about its target path (alignment gain) and about the concept it invents (invention gain),
using held-out language models as probability models on the KOMBINE analogy data.

## Setup

```bash
uv sync                                  # Python 3.12; mlx-lm for local scoring on Apple Silicon
# upstream data: ../comb-creat-eval/data/kg_creat/kombine_test30 (read-only)
# credentials for GPU runs: ../comb-creat-eval/.env (Lambda Cloud, HF token)
```

## Main scripts

```bash
bash scripts/analogy_ig/build_dataset.sh        # KOMBINE analogies + judge verdicts -> data/analogy_ig/dataset
bash scripts/analogy_ig/score_logprob.sh        # alignment and invention gains under one reader (config: model, backend)
bash scripts/analogy_ig/analyze_logprob.sh      # predictions and contrasts for one run
bash scripts/analogy_ig/analyze_models.sh       # cross-reader comparison table
bash scripts/analogy_ig/make_camera_figures.sh  # paper histograms (Nimbus Roman, ICML widths)
bash scripts/analogy_ig/lambda_run.sh <ip> ...  # run readers on a Lambda box; lambda_autofinish*.sh fetch and terminate
```

Never pass the overwrite flag to any script; choose a new `output_dir` instead.
