# 🧾 STS Evaluation Results Summary

This report compares all models across evaluation metrics (BERT, USE, LaBSE, LASER).
---

## 📊 average_bert_score

| Model                     |     Long |   Short_l6 |   Short_l6v2 |
|:--------------------------|---------:|-----------:|-------------:|
| Krutrim-2-instruct        | 0.744718 |   0.741878 |     0.715437 |
| Llama-3.1-8B-Instruct     | 0.699485 |   0.692418 |     0.670395 |
| OpenHathi-7B-Hi-v0.1-Base | 0.588734 |   0.55379  |     0.511845 |
| Qwen2.5-7B-Instruct       | 0.63038  |   0.625168 |     0.593522 |
| aya-23-8B                 | 0.667212 |   0.665826 |     0.635427 |
| sarvam-1                  | 0.616651 |   0.607249 |     0.565699 |


## 📊 average_use_score

| Model                     |     Long |   Short_l6 |   Short_l6v2 |
|:--------------------------|---------:|-----------:|-------------:|
| Krutrim-2-instruct        | 0.77527  |   0.756396 |     0.729685 |
| Llama-3.1-8B-Instruct     | 0.769663 |   0.764901 |     0.732444 |
| OpenHathi-7B-Hi-v0.1-Base | 0.40116  |   0.396126 |     0.32851  |
| Qwen2.5-7B-Instruct       | 0.482302 |   0.47704  |     0.395968 |
| aya-23-8B                 | 0.680117 |   0.668343 |     0.572852 |
| sarvam-1                  | 0.544582 |   0.534771 |     0.457689 |


## 📊 average_labse_score

| Model                     |     Long |   Short_l6 |   Short_l6v2 |
|:--------------------------|---------:|-----------:|-------------:|
| Krutrim-2-instruct        | 0.747034 |   0.726588 |     0.658079 |
| Llama-3.1-8B-Instruct     | 0.726546 |   0.710076 |     0.643619 |
| OpenHathi-7B-Hi-v0.1-Base | 0.481167 |   0.483612 |     0.381464 |
| Qwen2.5-7B-Instruct       | 0.60452  |   0.584343 |     0.496892 |
| aya-23-8B                 | 0.59104  |   0.577793 |     0.469548 |
| sarvam-1                  | 0.524186 |   0.516133 |     0.369713 |


## 📊 average_laser_score

| Model                     |     Long |   Short_l6 |   Short_l6v2 |
|:--------------------------|---------:|-----------:|-------------:|
| Krutrim-2-instruct        | 0.778855 |   0.769816 |     0.733646 |
| Llama-3.1-8B-Instruct     | 0.80547  |   0.799581 |     0.770166 |
| OpenHathi-7B-Hi-v0.1-Base | 0.587828 |   0.566241 |     0.493147 |
| Qwen2.5-7B-Instruct       | 0.760301 |   0.758068 |     0.743543 |
| aya-23-8B                 | 0.744403 |   0.738302 |     0.679488 |
| sarvam-1                  | 0.692623 |   0.69093  |     0.605618 |


## 🏁 Overall Average (Mean of BERT, USE, LaBSE, LASER)

| Model                     |     Long |   Short_l6 |   Short_l6v2 |
|:--------------------------|---------:|-----------:|-------------:|
| Krutrim-2-instruct        | 0.761469 |   0.74867  |     0.709211 |
| Llama-3.1-8B-Instruct     | 0.750291 |   0.741744 |     0.704156 |
| OpenHathi-7B-Hi-v0.1-Base | 0.514722 |   0.499942 |     0.428742 |
| Qwen2.5-7B-Instruct       | 0.619376 |   0.611155 |     0.557481 |
| aya-23-8B                 | 0.670693 |   0.662566 |     0.589329 |
| sarvam-1                  | 0.59451  |   0.587271 |     0.49968  |

