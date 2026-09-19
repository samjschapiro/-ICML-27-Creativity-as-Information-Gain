# Overwrite hazard: rebuilding an upstream destroyed its downstream runs

2026-09-18. `build_dataset.py --overwrite` called `init_directory`, which `rmtree`d
`data/analogy_ig/dataset/` including `downstream/`, where two log-prob runs lived (one complete,
2 h of compute; one in progress, whose open file handle then pointed at an unlinked inode). Both
were lost. Report numbers and figures survived because they had been copied into `docs/reports/`.

Fix: `src/utils.init_directory` now preserves a `downstream/` subdirectory on overwrite and prints
a notice. Rule of thumb: the dataset extraction is cheap and deterministic, the scoring is not;
anything expensive must live where a cheap rebuild cannot reach it.
