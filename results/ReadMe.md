# Experimental Results and Setup

This section documents the quantitative results, experimental setup, and implementation details of our project.

---

## Results Summary

We evaluate three model variants — **Long**, **Short_l6**, and **Short_l6v2** — under both **Base** and **Fine-tuned** settings. Performance is reported using two metrics:

* **Absolute**: Total number of correct predictions across the full evaluation set.
* **Same Samples**: Number of correct predictions restricted to a common subset of samples shared across all models, enabling a fair comparison.

### Fine-tuned Models

| Model Variant | Absolute | Same Samples |
| ------------- | -------- | ------------ |
| Long          | 902      | 390          |
| Short_l6      | 21388    | 390          |
| Short_l6v2    | 19746    | 390          |

### Base Models

| Model Variant | Absolute | Same Samples |
| ------------- | -------- | ------------ |
| Long          | 785      | 245          |
| Short_l6      | 19574    | 245          |
| Short_l6v2    | 18226    | 245          |


---

## Dataset Split Strategy

* **Training set:** 70%
* **Validation set:** 10%
* **Test set:** 20%

### Splitting Procedure

1. All available samples were loaded into memory.
2. The dataset was randomly shuffled using `random.shuffle()`.
3. A train/validation/test partition was created using a standard `train_test_split`-based procedure.

> **Note:** No fixed random seed was set during shuffling or splitting. As a result, exact reproducibility of the splits is not guaranteed. We therefore avoid making claims dependent on deterministic data partitioning. Model checkpoints are released publicly to support transparency and further analysis.

---

## Training Configuration

### Fine-tuning Time (Approximate)

| Model Variant | Fine-tuning Time |
| ------------- | ---------------- |
| Long          | ~4 hours         |
| Short_l6      | ~2.5 hours       |
| Short_l6v2    | ~2.5 hours       |


### Hardware Used

* **Fine-tuning:** Single 48 GB GPU
* **Inference:** 11 GB / 24 GB GPUs
* Multiple inference programs were executed in parallel.

---

## Inference Performance

| Model Variant | Inference Time |
| ------------- | -------------- |
| Long          | ~11 hours      |
| Short_l6      | ~6 hours       |
| Short_l6v2    | ~6 hours       |

These timings correspond to full evaluation runs over the respective test sets.

---

## Reproducibility Notes

* No explicit random seed was set during dataset shuffling or splitting.
* While this limits strict reproducibility, we mitigate this by releasing trained checkpoints publicly for verification and reuse.

Future iterations of this work will incorporate fixed random seeds and fully logged training configurations to ensure deterministic reproducibility.

---