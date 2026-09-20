# Project Structure

```
src/
├── utils.py                          # cross-track: load_config, init_directory (keeps downstream/ on overwrite), copy_config
└── analogy_ig/                       # track: information gain licensed by analogy, estimated with LM log-probs
    ├── dataset.py                    # join KOMBINE responses + judge verdicts + surprise -> analogy records
    ├── render.py                     # every prompt: alignment contexts, invention contexts, controls, direction detection
    ├── logprob.py                    # teacher-forced log-probs, mlx backend (local)
    ├── logprob_hf.py                 # same, transformers/CUDA backend (Lambda)
    └── scripts/
        ├── build_dataset.py          # -> data/analogy_ig/dataset/analogies.jsonl
        ├── score_logprob.py          # alignment + invention conditions per record -> downstream/logprob_<reader>_projdir/
        ├── score_format_control.py   # extra target-path controls (format | named) -> <run>/downstream/<control>_control/
        ├── score_invention_block.py  # invention as a skeleton task, whole description scored under four inputs -> <run>/downstream/invention_block/
        ├── analyze_logprob.py        # contrasts for one run
        ├── analyze_distance.py       # gain vs surprise and anchor distance
        ├── analyze_family.py         # gain by generating family
        ├── analyze_model.py          # gain by generating model, vs KOMBINE composite
        ├── analyze_models.py         # cross-reader comparison table
        ├── analyze_invention_block.py # cross-reader ladder for the invention block -> downstream/invention_block_ladder/
        ├── make_report_assets.py     # exploratory figures + rank-selected examples for the report
        ├── make_camera_figures.py    # paper histograms (PDF/PNG)
        ├── make_simple_figure.py     # one-example LaTeX figure (no longer used in the paper)
        └── make_examples_figure.py   # concrete-examples LaTeX figure (no longer used in the paper)

└── blend_ig/                         # track: information gain licensed by a blend (skeleton task, six-input ladder); reuses analogy_ig scorers
    ├── dataset.py                    # KOMBINE blends (concept, generic space, tagged triples) + panel verdicts -> V_bl
    ├── render.py                     # blend task, six appended inputs, per-triple span labels
    └── scripts/                      # build_dataset.py, score_blend_block.py, analyze_blend_block.py

configs/blend_ig/                     # build_dataset, analyze_blend_block, local/ (qwen7b mlx), models/ (five GPU readers)
scripts/blend_ig/                     # wrappers; lambda_run.sh, lambda_autofinish_blend.sh
data/blend_ig/dataset/                # blends.jsonl + downstream/ runs (gitignored)

configs/analogy_ig/                   # one YAML per script; models/ (per-reader scoring), format_control{,_local}/ (controls), invention_block{,_local}/
scripts/analogy_ig/                   # bash wrappers; lambda_run.sh, lambda_autofinish{,_controls,_invblock}.sh
data/analogy_ig/dataset/              # analogies.jsonl + downstream/ runs (gitignored)
logs/                                 # local run logs, Lambda launch/watcher logs (gitignored)

docs/
├── research_context.md               # theory bridge and current state
├── memos/                            # analogy gain taxonomy; overwrite hazard
├── reports/2026-09-18_analogy_ig_phase1/   # phase-1 report, figures, examples
├── tracks/analogy_ig/                # design.md (predictions, phases), progress.md (dated history)
├── tracks/blend_ig/                  # design.md (blend ladder, predictions fixed before scoring)
├── logs/                             # session logs
└── custom/paper_writing.md           # Overleaf conventions

literature/elmoznino_2024_compositionality/   # paper.pdf + paper.txt (gitignored)
papers/6aad5b22035168dccf225738/              # Overleaf clone (gitignored): setup/, sections/NN_*.tex, media/
```

Upstream data lives in `../comb-creat-eval/data/kg_creat/kombine_test30/` and is read-only here.
