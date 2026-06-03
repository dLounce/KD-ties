# KD-TIES: Cross-Manifold LoRA Merging in Knowledge Distillation

> **Diagnosing and correcting TIES merging failure modes in distillation-trained LoRA adapters for Qwen2.5-7B**





## What This Work Does

This project investigates whether TIES merging can compose instruction-following and mathematical reasoning capabilities when LoRA adapters are trained via knowledge distillation from a Qwen2.5-32B teacher into a Qwen2.5-7B student.

**The answer is no — and this work explains exactly why, and proposes two geometric corrections.**



## Key Findings

| # | Finding | Evidence |
|---|---|---|
| F1 | Distillation overwrites capability **before any merge** | Individual math LoRAs: GSM8K 0.466 vs baseline 0.642 |
| F2 | Cross-manifold task vectors collapse to **zero sign conflict** | CM-TIES = CM-Avg numerically across all benchmarks |
| F3 | TIES never significantly outperforms averaging in standard configs | No pairwise delta exceeds 2×SE on any benchmark |
| F4 | **SC-TIES** achieves only statistically significant cross-method gain | ΔMMLU = +0.017 > threshold 0.013 |
| F5 | **OP-TIES** achieves best commonsense retention | HellaSwag 0.792 · ρ₁₆ → 0.000 |
| F6 | Multi-adapter merging acts as implicit regulariser | CMAM GSM8K 0.514 vs individual floor 0.466 |

---

## Results

| Method | MMLU | HellaSwag | GSM8K | Sign Conflict |
|--------|------|-----------|-------|---------------|
| Baseline | 0.713 | 0.786 | 0.642 | — |
| SM-TIES | 0.699 ★ | 0.789 | 0.490 ★ | 0.221 |
| SM-Avg | 0.696 ★ | 0.787 | 0.468 ★ | — |
| CM-TIES | 0.706 | 0.781 | 0.480 ★ | 0.000 |
| CMAM | 0.705 | 0.790 | 0.514 ★ | 0.000 |
| CM-Avg | 0.706 | 0.781 | 0.486 ★ | — |
| **SC-TIES** | **0.716** | 0.786 | 0.452 ★ | 0.237 |
| **OP-TIES** | 0.711 | **0.792** | 0.484 ★ | 0.414 |

★ = statistically significant vs baseline (diff > 2×SE)

**SC-TIES** and **OP-TIES** are the two proposed methods.

---

## The Core Geometric Problem

Cross-manifold task vectors are computed as:

```
τ = θ_LoRA_on_TARGET − θ_BASE
```

This difference contains the entire instruct-tuning shift:

```
τ_anchor = θ_inst − θ_base
```

`τ_anchor` is orders of magnitude larger than the LoRA signal, making both task vectors point in nearly the same direction — producing **zero sign conflict**. TIES sign election has nothing to resolve and degenerates silently to plain averaging.

### SC-TIES Fix

```
τ_corrected = τ_cross − τ_anchor     # subtract anchor
τ_merged    = TIES(τ_corrected)      # merge residuals
θ_final     = θ_base + τ_anchor + τ_merged  # restore anchor
```

Sign conflict restored: **0.000 → 0.237**

### OP-TIES Fix

```
ΔW_projected = P_L · ΔW · P_R
where P_L = I − U₁₆U₁₆ᵀ  and  P_R = I − V₁₆V₁₆ᵀ
```

Subspace interference: **ρ₁₆: 0.026 → 0.000**

---

## Architecture

```
Teacher:        Qwen2.5-32B-Instruct  (generates distillation data)
Student Target: Qwen2.5-7B-Instruct   (evaluation baseline)
Student Base:   Qwen2.5-7B            (cross-manifold training base)

Domains:
  D1 — Instruction following  (GPT-4 dataset, 5,000 samples)
  D2 — Math reasoning         (MetaMathQA GSM_AnsAug, 5,000 samples)

LoRA Config:
  rank r = 16  |  alpha α = 16  |  dropout = 0.0
  target modules: q, k, v, o, gate, up, down (all 7)
  training: 1 epoch  |  lr = 2e-4  |  batch = 2  |  grad_accum = 4

TIES Config:
  retention density ρ = 0.70
  sign election: sum of signed magnitudes
  merge: mean of agreeing parameters only
```

---

## Pipeline

```
Phase 1 — Data Generation
  vLLM batch inference from 32B teacher
  temperature=0.0 (greedy, deterministic)
  ~15 minutes per domain on A100

Phase 2 — LoRA Training          ← KERNEL RESTART REQUIRED
  Unsloth supervised fine-tuning
  4 adapters: B1, B2 (same-manifold) · C1, C2 (cross-manifold)
  ~170MB per adapter checkpoint

Phase 3 — Merge and Evaluation   ← KERNEL RESTART REQUIRED
  In-memory TIES merge (never written to disk)
  7 variants evaluated sequentially
  Results saved to results.json after each variant

Phase 4 — Subspace Analysis
  ρk metric for k ∈ {8, 16, 32, 64, 128}
  56 layers × 4 adapters
  float32 precision mandatory for SVD stability
```

---

## Execution Environment

```
GPU:    A100 (102GB VRAM)
RAM:    30GB CPU RAM for task vector storage
Disk:   20GB working limit (Kaggle)

CRITICAL: vLLM and Unsloth cannot share a Python session.
Phase 1-2 and Phase 3-4 require separate kernel sessions.
```

To reproduce on Kaggle:
1. Enable GPU accelerator (A100)
2. Run Phase 1–2 cells → restart kernel
3. Run Phase 3–4 cells

---

## Repository Structure

```
kd-ties/
├── README.md
├── kd_ties.ipynb          ← full pipeline notebook
├── report/
│   └── kd_ties_thesis.pdf
├── results/
│   └── results.json
└── requirements.txt
```

---

## Requirements

```
torch>=2.0.0
transformers>=4.40.0
peft>=0.10.0
datasets>=2.18.0
vllm==0.4.2
unsloth
tqdm
numpy
scipy
```

---

## Thesis

Full thesis PDF available in `/report/kd_ties_thesis.pdf`

**Title:** Cross-Manifold Task Vectors and TIES Degeneration in Knowledge Distillation Pipelines

**Institution:** Central University of Rajasthan, Department of Computer Science

---

## Citation

```bibtex
@mastersthesis{raj2026kdties,
  author      = {Rishav Raj},
  title       = {Cross-Manifold Task Vectors and TIES Degeneration
                 in Knowledge Distillation Pipelines},
  school      = {Central University of Rajasthan},
  year        = {2026},
  month       = {June},
  department  = {Department of Computer Science}
}
```

---

## Acknowledgements

Supervised by Dr. Gaurav Meena, Department of Computer Science,
Central University of Rajasthan.
