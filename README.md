# music-plagiarism-detection

[![CI](https://github.com/MaksymSokolnytskyi/music-plagiarism-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/MaksymSokolnytskyi/music-plagiarism-detection/actions/workflows/ci.yml)

Interpretable similarity measures for musical borrowing, built from signal processing
rather than learned embeddings:

* constant-Q transform;
* key-aligned chroma with dynamic time warping;
* Markov chains over a jointly fitted NMF dictionary;
* self-similarity structure;
* a Fourier–Mellin invariant.

Every number in this README comes from a committed per-pair score file and the script
that produced it.

This started as a solo hackathon project (working name *MathDNA*) and went through four
versions: a math-heavy proof of concept, an over-engineered MVP, a Siamese CNN, and a
multi-chunk CNN. This repository is the rebuild. It keeps the mathematics, fixes three
bugs that had been silently degrading it, drops what did not work, and re-measures
everything — including the hackathon's own claims.

## Results

The short version:
- On clean or hand-cut material, the mathematics works.
- On full-length commercial recordings, nothing here — the hackathon's CNN included —
  separates borrowed from unrelated songs better than chance.

The tables below show both halves, and [docs/HISTORY.md](docs/HISTORY.md) explains why.

**1. Real excerpts: 27 unseen pairs, the same ones the CNN was tested on.** There are
11 derived pairs (plagiarism cases and remakes, e.g. *Under Pressure → Ice Ice Baby*,
*Without You*, Badfinger → Mariah Carey) and 16 unrelated pairings. All are 9–53 s
excerpts hand-cut around the borrowed passage. Nothing was tuned on these pairs.

| score | AUC | 95% CI |
|---|---|---|
| v3 CNN, embedding cosine | 0.47 | [0.23, 0.69] |
| v3 CNN, classifier head | 0.59 | [0.36, 0.81] |
| **key-aligned chroma DTW** | **0.75** | [0.53, 0.94] |
| **Markov chain over joint-NMF atoms** | **0.71** | [0.49, 0.90] |
| global chroma profile | 0.65 | [0.41, 0.88] |
| Fourier–Mellin descriptor | 0.59 | [0.37, 0.80] |
| self-similarity (form) | 0.52 | [0.27, 0.76] |

The order-aware harmonic features point the right way where the CNN did not. But this
is a direction with 11 positives, not a significant result: after a Bonferroni
correction across five features, no single p-value clears the 0.010 level.
[Full report](results/real/cnn-unseen/REPORT.md).

**2. Real full-length tracks: all 90 pairs of the v4 manifest.** There are 43 covers,
remakes and borrowings and 47 unrelated pairs; median length is 4 minutes.

| score | AUC | 95% CI |
|---|---|---|
| global chroma profile | 0.58 | [0.46, 0.70] |
| key-aligned chroma DTW | 0.53 | [0.41, 0.65] |
| Markov chain over joint-NMF atoms | 0.58 | [0.46, 0.70] |
| self-similarity (form) | 0.53 | [0.42, 0.65] |
| Fourier–Mellin descriptor | 0.60 | [0.48, 0.71] |
| *exploratory:* 30 s window search, all keys | 0.50 | [0.39, 0.62] |

Every interval contains or nearly touches 0.5. The positive and negative score
distributions are almost the same: the mean global chroma similarity is 0.970 for
covers and 0.965 for unrelated songs. Full-mix chroma of Western pop and rock saturates
— every song resembles every other.

The window search tested one explanation: that global alignment drowns a borrowed
passage. It was designed after seeing the first table, and it did not help (0.50).
[Full report](results/real/v4/REPORT.md).

**3. Controlled synthetic edits: 120 pairs with exactly known ground truth.** Suspects
are transposed, tempo-shifted, re-orchestrated, given random intros and mixed with
noise. The hard negatives share the original's key, tempo, chords and sound, and differ
only in the melody.

* Key-aligned chroma DTW reaches AUC 0.94 overall, and 0.86–1.00 against the hard
  negatives.
* The window search reaches 0.996.
* The joint-NMF fix lifts DTW on activations from 0.37 to 0.68 overall, and from 0.49
  to 0.99 for covers vs unrelated songs.
* Transposition is recovered exactly in 93% of positive pairs.

[Full report](results/synthetic/REPORT.md).

Read together, the three tables make one finding. The transforms do what the
mathematics says: the synthetic benchmark and the tests show it. On real full mixes,
the *representation* — 12-bin chroma of the whole mix — no longer carries the
difference between "borrowed" and "same genre". That points to melody extraction
(source separation, then a pitch contour) rather than a better classifier.

**The hackathon's own numbers, recomputed** in
[results/history/REPORT.md](results/history/REPORT.md):

| version | best honest number | what it means |
|---|---|---|
| v1 PoC | AUC 0.71 [0.56, 0.84] for its DTW+Markov channel | 16 curated positives vs 50 *random* pairings — easy negatives |
| v2 MVP | — | claimed AUC 0.91; no per-pair scores or logs exist, so not used |
| v3 CNN | AUC 0.47 on unseen tracks | maps every pair of songs to ~0.85 similarity |
| v4 | best validation loss 0.674 vs 0.690 for a constant | memorisation; split verified leak-free |

## What changed from the hackathon code

The full story is in [docs/HISTORY.md](docs/HISTORY.md). In short:

1. **NMF atoms were compared across separately fitted factorisations.** NMF is unique
   only up to a permutation of atoms, so "atom 3" of song A and of song B were
   unrelated objects. One dictionary is now fitted on both recordings (`nmf.joint_nmf`).
   The tests show two fits of the *same* matrix disagreeing on most frame labels.
2. **The log-polar warp encoded the wrong invariances.** A tempo change scales one axis
   and a transposition shifts the other. Isotropic log-polar coordinates suit neither.
   It is replaced by a one-axis Fourier–Mellin descriptor (`invariants`).
3. **The DTW band made tempo changes unreachable.** The fixed band $|i-j|\le W$ returns
   $\infty$ once the lengths differ by more than $W$. The band now follows the
   rectangle's diagonal.
4. Flattened, asymmetric KL between Markov chains became occupancy-weighted
   Jensen–Shannon divergence.
5. A constant LPC channel (~96% on random pairs) was dropped.
6. A deep-learning branch that never generalised was dropped, and its failure is
   documented.

## The mathematics

```
audio ─► CQT ─┬─► chroma ─► optimal transposition ─► DTW        order of harmony/melody, tempo-free
              │                                    └─► SSM       song form
              ├─► joint NMF ─► atom states ─► Markov + JS        how the harmony moves
              └─► |FFT2| ─► log modulation axis ─► |FFT|         transposition- and tempo-invariant texture
```

Derivations, references and the reasons behind each choice are in
[docs/MATH.md](docs/MATH.md). The tests check the mathematics against analytic facts:

* a pure tone lands in its semitone bin;
* transposition is a bin shift;
* DTW equals the textbook recursion and costs zero on an exact stretch;
* subsequence DTW finds an embedded quote's span;
* KL-NMF updates never increase the divergence;
* the Tonnetz matches librosa to $10^{-6}$;
* the AUC matches scikit-learn with ties;
* the descriptor is exactly invariant to circular shifts.

## Use

```bash
pip install -e ".[audio]"              # numpy, scipy, pydantic + librosa for decoding
musicplag original.wav suspect.mp3     # JSON: feature values, key offset, alignment evidence
```

```python
from musicplag import PairContext, analyze_pair

report = analyze_pair(PairContext.from_audio(y_original, y_suspect))
report.features                  # {"chroma_dtw": ..., "markov_js": ..., ...}
report.transposition_semitones   # e.g. +3: the suspect is three semitones higher
```

The output is evidence, not a verdict. No threshold here has been validated on enough
real pairs to call anything plagiarism.

## Reproduce

```bash
pip install -e ".[dev,audio]"
pytest                                          # 53 tests, ~10 s
python experiments/reanalyze_history.py         # results/history/REPORT.md (CI checks it is unchanged)
python experiments/synthetic_benchmark.py       # results/synthetic/, ~4 min
python experiments/real_pairs.py --pairs cnn-unseen --audio-dir DIR   # results/real/cnn-unseen/
python experiments/real_pairs.py --pairs v4 --audio-dir DIR           # results/real/v4/, ~25 min
```

Audio and model weights are not in the repository; the recordings are copyrighted.
Pair manifests and every per-pair score are.

## Layout

```
src/musicplag/   spectral · dtw · nmf · markov · structure · invariants   (mathematics)
                 features (pair analysis) · metrics · data · synth · cli
experiments/     one script per question, each writing a REPORT.md next to its scores
results/         history/ (v1–v4 re-analysis) · synthetic/ · real/cnn-unseen/ · real/v4/
docs/            MATH.md · HISTORY.md
```

## Limitations

* **Small real pair sets.** 27 excerpt pairs and 90 full-track pairs give wide
  intervals. They are enough to tell "works" from "doesn't", not to rank similar
  features.
* **Chroma is the bottleneck on real mixes.** On the synthetic "same chord loop"
  negatives, the Markov and Fourier–Mellin features prefer the shared loop to the
  borrowed melody (AUC < 0.5). That is legally the wrong way round, and the full-track
  table suggests real recordings behave the same way. The missing pieces are melody
  extraction (source separation, then a pitch contour), beat-synchronous features, and
  cross-recurrence alignment of the kind used in cover-song identification.
* **One post-hoc probe.** The window search was designed after seeing the full-track
  table, and it is labelled exploratory everywhere it appears. Nothing else was
  changed in response to a real-data result. Two bugs found by sanity checks (silent
  frames, CQT speed) were fixed, and every table was regenerated afterwards.
* **Synthetic audio is easy.** Exact pitches and no vocals, drums or mixing. Passing
  that benchmark is necessary, not sufficient.

## License

MIT for the code. The track titles in `results/` identify recordings owned by their
rights holders and are listed for reproducibility only.
