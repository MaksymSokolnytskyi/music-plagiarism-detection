# How this got here: v1 → v4

A solo hackathon project, written over about two weeks in January–February 2026 under
the working name *MathDNA*. It went through four rewrites. Each one was a reaction to
what the previous one measured. This file records what each version tried, what it
measured, and what this repository keeps. All numbers are recomputed from the
per-pair scores each version saved ([results/history/REPORT.md](../results/history/REPORT.md)).

| version | idea | what it measured | kept here |
|---|---|---|---|
| v1 PoC | CQT → per-track NMF; a "three-pillar" score (DTW on activations, log-polar patch correlation, Markov/KL on atoms) + LPC timbre | AUC 0.61 for the combined score on 16 known vs 50 random pairs. The DTW+Markov channel alone reached **0.71** [0.56, 0.84], the best historical signal | CQT, DTW, NMF, Markov — rebuilt with three bugs fixed |
| v2 MVP | Seven "processors" (harmonic, Tonnetz, rhythm, SSM, melodic n-grams, persistent homology, log-polar), deep verification (DTW + SSM), evidence chain, HTML reports | Claimed AUC 0.91 / F1 84.6%. **No per-pair scores or logs exist**, and the feature aggregator returned a zero vector, so the claim is not reproducible | Tonnetz, SSM, the evidence idea |
| v3 CNN | Siamese VGGish embeddings (concat-diff head) + XGBoost over CNN and hand-made features; 90 training pairs, augmented to 3,843 | Unseen tracks: embedding cosine AUC **0.47**, classifier head 0.59 (CI includes 0.5). XGBoost put 100% importance on the CNN score | Kept as a documented negative result, not as code |
| v4 | Full-track multi-chunk scanning, CNN + interpretable features, track-disjoint splits | Splits verified leak-free. The best validation loss, 0.674, barely beats a constant predictor at 0.690; validation loss rose while training loss fell | Track-disjoint evaluation; chunking as future work |

## What went wrong, specifically

These are the mechanisms behind the numbers above, found while rebuilding. Each is
either fixed or dropped in this repository.

1. **Per-track NMF gave atoms arbitrary labels (v1–v3).** NMF is identifiable only up to
   a permutation, so comparing $H_A$ and $H_B$ row by row compared unrelated atoms.
   Fixed with a shared dictionary (`nmf.joint_nmf`). On the synthetic benchmark the
   same DTW goes from AUC 0.37 to 0.68 overall, and from 0.49 to 0.99 on covers vs
   unrelated songs — see [results/synthetic/REPORT.md](../results/synthetic/REPORT.md).
2. **The log-polar warp encoded the wrong invariances (v1–v2).** An image log-polar
   transform makes isotropic scaling and rotation into shifts. Musical edits are a
   one-axis time scaling and a pitch-axis *shift*. Replaced by a one-axis
   Fourier–Mellin descriptor (`invariants`).
3. **The DTW band broke on tempo changes (v4).** The constraint $|i-j|\le W$ makes the
   end cell unreachable when the lengths differ by more than $W$. The band now follows
   the rectangle's diagonal.
4. **Markov chains were compared by flattened KL (v1).** KL is asymmetric and
   unbounded, and flattening weights never-visited states like the chorus chord.
   Replaced by occupancy-weighted Jensen–Shannon.
5. **One channel was constant (v1).** LPC timbre scored ~96% on random pairs. Dropped.
6. **A calibration file reported a false-positive rate of 2.0 (v1).** That is
   impossible for a rate. The PoC metrics are recomputed here from raw scores instead.
7. **A headline number had no data behind it (v2).** It is not used.
8. **The network learned "these are both songs" (v3).** Cosine similarity was ~0.85
   for every pair. With 90 pairs, augmentation multiplies examples of the *same
   recordings*, not of borrowing.

## What the hackathon got right

* **The hypothesis.** Plagiarism is a relation between musical content, so it should
  be detected by functions with the right invariances. That is the direction this
  repository takes.
* **Explainable evidence over a black-box score.** The v2 `EvidenceChain` became
  `PairReport`.
* **Track-disjoint splits in v4.** They are verified here, and they are why the v3/v4
  failures are trustworthy as failures.
* **The DTW+Markov channel of v1** was the one historical signal clearly above chance,
  despite the NMF labelling bug.

## Not reproduced

Persistent homology (v2) and the FAISS/HNSW index with a provenance graph (v1–v2) were
engineering for a catalogue-scale system. Without a working pairwise comparison there
was nothing for them to scale. The CNN weights (220 MB) and all audio stay out of the
repository.

## Where the rebuild ended up

The rebuilt features are tested two ways. On the 27 hand-cut excerpts they point the
right way where the CNN did not (chroma DTW 0.75 vs CNN 0.47–0.59). On the 90
full-length pairs they sit at chance, like everything before them. The synthetic
benchmark shows the transforms behave as derived. So the limit is the input
representation — full-mix chroma — and not the alignment mathematics. Two bugs were
found by sanity checks during this evaluation and fixed before the final tables:

* silent frames had a cosine distance of 1 to themselves, so an identical pair scored
  0.981;
* the direct CQT was too slow for full tracks, so a multirate version replaced it.

