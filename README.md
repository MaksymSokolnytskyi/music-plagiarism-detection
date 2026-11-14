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

**On real recordings: 27 unseen pairs, the same ones the CNN was tested on.** There are
11 derived pairs (plagiarism cases and remakes, e.g. *Under Pressure → Ice Ice Baby*,
*Without You*, Badfinger → Mariah Carey) and 16 unrelated pairings, all curated excerpts.
Nothing was tuned on these pairs.

| score | AUC | 95% CI |
|---|---|---|
| v3 CNN, embedding cosine | 0.47 | [0.23, 0.69] |
| v3 CNN, classifier head | 0.59 | [0.36, 0.81] |
| **key-aligned chroma DTW** | **0.76** | [0.54, 0.94] |
| **Markov chain over joint-NMF atoms** | **0.76** | [0.56, 0.92] |
| Fourier–Mellin descriptor | 0.60 | [0.39, 0.81] |
| global chroma profile | 0.67 | [0.44, 0.89] |
| self-similarity (form) | 0.56 | [0.31, 0.78] |

The two order-aware harmonic features point the right way where the CNN did not. With
11 positives that is a direction, not a result: after a Bonferroni correction across
the five features, no single p-value is significant. Full table:
[results/real/REPORT.md](results/real/REPORT.md).

**On controlled synthetic edits: 120 pairs with exactly known ground truth.** Suspects
are transposed, tempo-shifted, re-orchestrated, given random intros and mixed with
noise. The hard negatives share the original's key, tempo, chords and sound, and differ
only in the melody. Key-aligned chroma DTW separates derived from unrelated pairs at
AUC 0.93, and at 0.86–1.00 against those hard negatives. The joint-NMF fix lifts the
same DTW from 0.39 to 0.67 overall, and from 0.50 to 0.99 for covers. Transposition is
recovered exactly in 93% of positive pairs. See
[results/synthetic/REPORT.md](results/synthetic/REPORT.md).

**The hackathon's own numbers, recomputed** in
[results/history/REPORT.md](results/history/REPORT.md):

| version | best honest number | what it means |
|---|---|---|
| v1 PoC | AUC 0.71 [0.56, 0.84] for its DTW+Markov channel | the one real early signal, despite an NMF labelling bug |
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
report.features                  # {"chroma_dtw": 0.91, "markov_js": 0.88, ...}
report.transposition_semitones   # e.g. +3: the suspect is three semitones higher
```

The output is evidence, not a verdict. No threshold here has been validated on enough
real pairs to call anything plagiarism.

## Reproduce

```bash
pip install -e ".[dev,audio]"
pytest                                          # 50 tests, ~10 s
python experiments/reanalyze_history.py         # results/history/REPORT.md (CI checks it is unchanged)
python experiments/synthetic_benchmark.py       # results/synthetic/, ~4 min
python experiments/real_pairs.py --audio-dir DIR   # results/real/, needs the audio excerpts
```

Audio and model weights are not in the repository; the recordings are copyrighted.
Pair manifests and every per-pair score are.

## Layout

```
src/musicplag/   spectral · dtw · nmf · markov · structure · invariants   (mathematics)
                 features (pair analysis) · metrics · data · synth · cli
experiments/     one script per question, each writing a REPORT.md next to its scores
results/         history/ (v1–v4 re-analysis) · synthetic/ · real/
docs/            MATH.md · HISTORY.md
```

## Limitations

* **27 real pairs.** That is enough to see that the CNN failed and to see a direction
  for the harmonic features. It is not enough to rank features or choose a threshold.
  The next step is the 90-pair, track-disjoint v4 manifest; the features need no
  training, so all of it can be test data.
* **Harmony is not melody.** Chroma mixes the vocal line with the accompaniment. On the
  synthetic "same chord loop" negatives, the Markov and Fourier–Mellin features prefer
  the shared loop to the borrowed melody (AUC < 0.5). That is legally the wrong way
  round. Melody extraction (source separation, then a pitch contour) is the missing
  piece.
* **Excerpts, not songs.** The real pairs are hand-cut around the borrowed passage.
  Finding the passage in full tracks is what subsequence DTW is for, but it is not yet
  evaluated.
* **Synthetic audio is easy.** Exact pitches and no vocals, drums or mixing. Passing
  that benchmark is necessary, not sufficient.

## License

MIT for the code. The track titles in `results/` identify recordings owned by their
rights holders and are listed for reproducibility only.
