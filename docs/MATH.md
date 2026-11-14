# The mathematics

Each section names the module that implements it. Module docstrings carry the same
formulas next to the code.

Musical borrowing nearly always comes with edits: the borrowed material is
**transposed**, played at another **tempo**, re-orchestrated (**timbre**), cut into
another **form**, or **quoted** inside a longer track. A useful similarity is invariant
to the edits that don't matter and sensitive to the content that does. The pipeline
below handles one edit per step.

```
audio ─► CQT (log-frequency) ─┬─► chroma ─► optimal transposition ─► DTW            (melody/harmony order, tempo-free)
                              │                                   └─► SSM           (song form)
                              ├─► joint NMF ─► atom states ─► Markov chain + JS     (harmonic motion)
                              └─► 2-D |FFT| ─► log modulation axis ─► |FFT|         (Fourier–Mellin invariant)
```

## 1. Constant-Q transform — `spectral.cqt`

Bin $k$ sits at $f_k = f_{\min}\,2^{k/b}$ with $b = 12$ bins per octave. The window
length $N_k = \lceil Q f_s / f_k\rceil$ keeps the quality factor
$Q = (2^{1/b}-1)^{-1} \approx 16.8$ fixed (Brown 1991):

$$X[k,n] = \frac{1}{N_k}\sum_{m=0}^{N_k-1} x\!\left[nH + m - \lfloor N_k/2\rfloor\right] w_{N_k}[m]\, e^{-j2\pi Qm/N_k}.$$

**The property everything else uses:** a transposition by $s$ semitones multiplies
every frequency by $2^{s/12}$, which on this axis is a *shift by $s$ bins*. Tests check
this on pure tones (`tests/test_spectral.py`).

## 2. Chroma, Tonnetz and the transposition estimate — `spectral`

Chroma folds octaves: $c[p,n] = \sum_o |X[p+12o,n]|^2$, normalised per frame. The
Tonnetz (Harte, Sandler & Gasser 2006) maps chroma onto three circles — fifths, minor
thirds and major thirds — $\zeta = \Phi c/\lVert c\rVert_1$, with
$\Phi_{2i,p} = r_i\sin(p\,\theta_i)$, $\Phi_{2i+1,p} = r_i\cos(p\,\theta_i)$ and
$\theta = (7\pi/6,\,3\pi/2,\,2\pi/3)$, $r = (1, 1, \tfrac12)$. It matches librosa to
$10^{-6}$.

The **optimal transposition index** (Serrà, Gómez & Herrera 2008) compares the global
pitch-class profiles under all 12 rotations:
$k^* = \arg\max_k \langle \bar c_A,\ \mathrm{roll}(\bar c_B,k)\rangle$. Every later
step compares $A$ with $B$ rotated by $k^*$, so a cover in another key costs nothing.
It also yields the first piece of evidence: "B is A transposed by +3 semitones".

## 3. Dynamic time warping — `dtw`

With cosine cost $C_{ij} = 1-\cos(a_i,b_j)$,

$$D_{ij} = C_{ij} + \min\{D_{i-1,j},\ D_{i,j-1},\ D_{i-1,j-1}\},$$

and the similarity is $1 - D_{NM}/L$ for a path of length $L$. A uniform tempo change
repeats or skips frames, which DTW absorbs at zero cost (tested on an exact 2×
stretch).

Two details differ from the hackathon code:

* **Band.** The Sakoe–Chiba constraint $|i-j|\le W$ makes the end cell unreachable as
  soon as $|N-M| > W$, so a 1.3× tempo change returned $+\infty$. The band here follows
  the diagonal of the rectangle: $|i/(N-1) - j/(M-1)| \le w$.
* **Speed.** Cells on an anti-diagonal $i+j=d$ depend only on earlier anti-diagonals,
  so each diagonal is one vectorised NumPy step. That is $O(N+M)$ Python iterations
  instead of $O(NM)$. It agrees exactly with the textbook recursion.

**Subsequence DTW** (Müller, *FMP* §7.2) sets $D_{0,j} = C_{0,j}$ (free start) and takes
$\min_j D_{N-1,j}$ (free end). That is the model for a sample or a quoted hook inside a
longer track; the test recovers an embedded quote's exact span.

## 4. NMF with a shared dictionary — `nmf`

$V\approx WH$, $W,H\ge 0$, minimising the generalised KL divergence with the Lee & Seung
(2001) multiplicative updates

$$H \leftarrow H\odot\frac{W^\top(V\oslash WH)}{W^\top\mathbf 1},\qquad
W \leftarrow W\odot\frac{(V\oslash WH)H^\top}{\mathbf 1H^\top}.$$

These updates never increase the divergence (their Theorem 2). The tests check that
the loss curve is monotone, and that an exactly low-rank matrix is recovered to 1%.

**The fix.** v1–v3 factorised each recording separately and then compared $H_A$ with
$H_B$ row by row (DTW) and as Markov states. NMF is unique only up to a permutation and
scaling of atoms, so "atom 3" in one recording and "atom 3" in the other are unrelated
objects. The tests show two NMF runs on *the same* matrix disagreeing on most frame
labels. Fitting one dictionary on $[V_A\ V_B]$ — after key alignment — makes a state
label mean the same chord or timbre in both recordings.

## 5. Markov chains over atoms — `markov`

States $s_t = \arg\max_k H_{kt}$; transitions $P_{ij} = \Pr(s_{t+1}=j\mid s_t=i)$ with
Lidstone smoothing. The chains are compared by the occupancy-weighted Jensen–Shannon
divergence

$$D = \sum_i \pi_i\,\mathrm{JS}(P_A[i], P_B[i]),\qquad
\mathrm{JS}(p,q)=\tfrac12 KL(p\|m)+\tfrac12 KL(q\|m),\ m=\tfrac{p+q}{2},$$

and $\text{similarity} = 1 - D/\ln 2 \in [0,1]$. JS is symmetric and bounded. The
weights $\pi$ stop rows of states that are never visited (pure smoothing) from counting
as much as the chorus chord, which they did in v1's flattened KL.

## 6. Self-similarity and form — `structure`

$S_{ij} = \cos(f_i, f_j)$ over chroma frames (Foote 1999). Repeated sections are
stripes parallel to the diagonal, and homogeneous sections are blocks. A uniform tempo
change only rescales the picture, so both SSMs are resampled to $48\times48$. They are
compared by Pearson correlation of the upper triangle, excluding a band of 2 cells
around the diagonal, which is trivially similar in every recording.

## 7. Fourier–Mellin invariant — `invariants`

On a CQT patch $X(f,t)$:

1. $|\mathcal F_2 X|$ is invariant to shifts in both axes, i.e. to transposition and to
   time offset (the shift theorem).
2. A tempo change $X(f, t/a)$ scales the temporal-modulation axis,
   $\omega_t\mapsto a\,\omega_t$. Resampling that axis logarithmically,
   $u = \ln\omega_t$ (in hertz, so different patch lengths are comparable), turns the
   scaling into a shift $u\mapsto u+\ln a$.
3. A second Fourier magnitude along $u$ removes that shift. This is the scale (Mellin)
   transform (Cohen 1993) applied to the time axis only.

v1 used an image log-polar warp (`cv2.logPolar`) around the centre of an STFT patch.
Log-polar coordinates turn *isotropic* scaling and *rotation* into shifts. Neither
matches a musical edit: tempo scales one axis, and transposition shifts rather than
scales on a log-frequency axis. The construction above encodes the two real edits. Step
1 is exact (tested to $10^{-10}$). Step 3 is approximate because the frequency range is
finite.

**Phase correlation** recovers a shift exactly:
$\arg\max \mathcal F^{-1}\{\hat a\hat b^*/|\hat a\hat b^*|\}$ (Kuglin & Hines 1975).

## 8. Evaluation statistics — `metrics`

* AUC is the Mann–Whitney statistic with midranks, so ties count ½. It is checked
  against scikit-learn.
* Uncertainty comes from a stratified percentile bootstrap: positives and negatives are
  resampled separately.
* The p-value is a one-sided permutation test of AUC = 0.5.
* The effect size is Cohen's $d$ with pooled SD.

With 14–66 pairs per experiment, a single accuracy at a tuned threshold is mostly noise.
That is why every number in this repository is an AUC with an interval.

## What was dropped, and why

* **LPC pole–zero timbre (v1).** It scored $96.3 \pm 1.9$ on *random* pairs: everything
  looked 96% alike (AUC 0.45).
* **Persistent homology (v2).** The Vietoris–Rips $H_1$ of chroma point clouds depended
  on `ripser` and fell back to a heuristic without it. It was never evaluated on its own
  and is not reproduced here.
* **The CNN (v3–v4).** See [HISTORY.md](HISTORY.md). It is kept as a documented
  negative result, not as code.

## References

Brown 1991 (CQT) · Foote 1999 (SSM) · Harte, Sandler & Gasser 2006 (Tonnetz) ·
Kuglin & Hines 1975 (phase correlation) · Lee & Seung 2001 (NMF updates) · Lin 1991
(Jensen–Shannon) · Müller, *Fundamentals of Music Processing*, 2015 (DTW, subsequence
DTW) · Sakoe & Chiba 1978 (DTW band) · Serrà, Gómez & Herrera 2008 (OTI) · Smaragdis
& Brown 2003 (NMF for music) · Cohen 1993 (scale transform).
