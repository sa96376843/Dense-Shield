# DenseShield Figure 1 regeneration prompt

Generation route: built-in GPT Image 2 workflow (`imagegen` skill), completed 2026-09-08.

Use case: scientific-educational. Asset type: final full-width IEEE TIFS framework figure, approximately 2.1:1 landscape.

Create a polished systems diagram for DenseShield on a clean white background using flat vector-like line art, an Okabe--Ito-inspired color-blind-safe palette, dark outlines, generous whitespace, and consistent line weights. Use solid arrows for the main data/decision path and dashed arrows for auxiliary, shared-representation, and model-update paths. Use two swimlanes: `OFFLINE TRAINING & PURIFICATION` and `ONLINE INFERENCE & ADAPTATION`.

Offline path: `Training data D_tr` -> `STAGE 1 Sparse Feature Compression (SFC)` (`rare features -> B hash bins`, `x-tilde = phi(x)`) -> `STAGE 2 Density-Boosted Adversarial Training (DBAT)` (`k-NN density weights`, `weighted min-max training`) -> `STAGE 3 D-PUR Backdoor Purification` (`saliency x rarity`, `counter-example unlearning`) -> `Robust classifier f_theta*`. Add `1% clean hold-out` as a dashed auxiliary input to Stage 3.

Online path: `Incoming sample x` -> `SFC mapping x-tilde = phi(x)` -> `Robust classifier f_theta*` -> `Prediction benign / malware`. Branch from the classifier to `STAGE 4 Uncertainty-Conformal Drift Detection` (`uncertainty u(x)`, `batch p-value p_t < alpha`) -> `drift alarm` -> `STAGE 5 Drift-adaptive Minimal-Label Adaptation (DMLA)` (`top-q analyst labels`, `confidence-gated pseudo-labels`, `replay regularization`). Add a dashed `model update` feedback arrow from Stage 5 to the online classifier. Connect the two SFC boxes with a dashed cue labeled `shared density-centric representation`.

Constraints: no fabricated plots or data, no extra stages, no gradients, no shadows, no watermark, no internal `Figure 1` title, no clipped text, and no crossing arrows. Render all labels exactly and keep them readable at final two-column width.

The selected GPT Image 2 composition is retained as
`fig1_framework_gpt_image2.png`, and the original reference is
`fig1_original.png`. Following the project rule that small components should be
code-generated whenever possible, the submission figure was subsequently
rebuilt from that composition with deterministic vector primitives. The final
outputs are `fig1_framework_hybrid.pdf` and `fig1_framework_hybrid.png`; their
editable generator is `generate_fig1_hybrid.py`.
