mean_global_importance_ad_{seed}.csv
mean_global_importance_dis_{seed}.csv

-> Graph based integrated gradients / num_paths
-> avoid value explosion based on num_paths

좋아. 지금 업로드한 **세 개의 파일**을 기준으로,
네가 만든 **GB-IG (Graph-Based Integrated Gradients) for CTMP-GIN** 파이프라인을 사용하는 방법을 정리한 **README 초안**을 만들어 줄게.

아래 내용은 실제 코드 흐름을 그대로 반영해서 작성했고,
특히 `gb_ig.py`와 `explainer_main.py`의 동작을 정확히 설명하도록 구성했어.
(코드 인용: , )

---

# 📘 GB-IG Explainer for CTMP-GIN

This repository provides a **Graph-Based Integrated Gradients (GB-IG)** implementation for interpreting **CTMP-GIN** models.
It follows the formulation in:

> *Graph-based Integrated Gradients for Explaining Graph Neural Networks*
> (Short Path Entropy Baseline + Path Attribution)

and includes:

* **Shortest-path entropy baseline selection**
* **Tensorized GPU path scoring**
* **Global importance aggregation**
* **Stability analysis across seeds**

---

## 1. Core Idea

For a target variable node ( v ), GB-IG computes:

[
\text{GBIG}(v) = \mathbb{E}*{\gamma \in \Gamma(b,v)}
\left[
\sum*{(u \to w) \in \gamma}
\left( X_w - X_u \right)^\top \nabla X_u
\right]
]

Where:

* ( \Gamma(b,v) ): shortest paths from baseline ( b ) to target ( v )
* ( X ): node embeddings
* ( \nabla X ): gradients of model output w.r.t embeddings
* ( b ): **entropy-maximizing farthest node** from ( v )

This is implemented in:

```python
def gbig_score_for_node_tensorized(...)
```



---

## 2. Baseline Selection (Entropy Criterion)

For each target node `v`, we select baseline `b`:

[
b = \arg\max_{u \in D(v)}
\sum_{\gamma \in \Gamma(u,v)} p(\gamma) I(\gamma)
]

Where:

[
p(\gamma) = \prod_{x \in \gamma} \frac{1}{\deg(x)},
\quad
I(\gamma) = \sum_{x \in \gamma} \log_2 \deg(x)
]

Implemented in:

```python
choose_baseline_node(...)
```



**Optimization:**
Only **one BFS from target** is used, and paths are enumerated in reverse (target → b).
Direction does **not matter** because the formula only uses node products/sums.

---

## 3. Path Caching

To avoid recomputing shortest paths every iteration, we cache:

```python
class ShortestPathEdgeCache:
```

For each target node `v`:

1. Select baseline `b`
2. BFS from `b`
3. Enumerate up to `max_paths` shortest paths
4. Convert each path into `(u_idx, w_idx)` edge pairs



This cache is built once in:

```python
self.path_cache.build()
```

---

## 4. Tensorized Path Attribution

Instead of looping over edges in Python, each path is scored on GPU:

```python
delta = X[w] - X[u]
edge_contrib = (delta * G[u]).sum(dim=-1)
path_score = edge_contrib.mean()
```

This is done in:

```python
gbig_score_for_node_tensorized(...)
```



You can choose:

* `use_mean=True` → average over edges and paths
* `use_mean=False` → sum (if you want path count to matter)

---

## 5. Explaining a Batch

```python
res = explainer.explain_batch(x, los, edge_index)
```

Internally:

1. Loop over samples in the batch
2. Forward 1 sample
3. Backprop scalar output
4. Capture embeddings + gradients via hook
5. For each variable node:

   * Load cached shortest paths
   * Compute GB-IG score

Implemented in:

```python
class CTMPGIN_GBIGExplainer
```



---

## 6. Global Importance (Dataset-Level)

```python
compute_global_importance_on_loader(...)
```

* Randomly samples batches (`sample_ratio`)
* Computes GB-IG per sample
* Aggregates with `mean` or `median`

Key line:

```python
global_var = running_sum_var / n_seen
```

Here, `n_seen` is the **total number of samples processed**, not batches.



---

## 7. Running the Explainer

### Step 1. Load model and data

```bash
python explainer_main.py --config configs/ctmp_gin.yaml
```

Model is loaded from:

```python
model_path = runs/.../ctmp_epoch_37_loss_0.2717.pth
```



---

### Step 2. Create Explainer

```python
explainer = CTMPGIN_GBIGExplainer(
    model=model,
    edge_index_vargraph=edge_index,
    ad_indices=dataset.col_info[2],
    dis_indices=dataset.col_info[3],
    baseline_strategy="farthest",
    max_paths=1,
    use_abs=True,
)
```

---

### Step 3. Global Importance + Stability

```python
out = compute_global_importance_on_loader(
    explainer,
    model,
    test_loader,
    edge_index,
    device,
    sample_ratio=0.05,
    seed=s,
)
```

Repeated across seeds → stability analysis.



---

## 8. Stability Analysis

From `stablity_report.py`:

* Top-K overlap
* Unstable variable detection
* Mean ± std tables

Used in:

```python
report(...)
```

---

## 9. Key Configuration Knobs

| Parameter             | Meaning                     |
| --------------------- | --------------------------- |
| `baseline_strategy`   | `"farthest"` or `"fixed"`   |
| `max_paths`           | max shortest paths per node |
| `use_mean_in_explain` | average vs sum over paths   |
| `sample_ratio`        | fraction of dataset used    |
| `reduce`              | `"mean"` or `"median"`      |

---

## 10. Notes

* Path direction does **not** matter for entropy baseline.
* Using `sum` instead of `mean` will make **path count influence importance**.
* Tensorization removes Python edge loops (major speedup).
* Baseline is chosen **per target variable**, not global.

---

If 원하면 다음 단계로:

* 논문 식과 네 코드 1:1 매핑
* Appendix 스타일 pseudo-code
* Reviewer 대응용 justification 문장

도 같이 만들어 줄게.
