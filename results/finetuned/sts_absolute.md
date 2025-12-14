# 🧾 STS Evaluation Results Summary

This report compares all models across evaluation metrics (BERT, USE, LaBSE, LASER).
---

## 📊 average_bert_score

| Model                     |     Long |   Short_l6 |   Short_l6v2 |
|:--------------------------|---------:|-----------:|-------------:|
| Krutrim-2-instruct        | 0.740126 |   0.729732 |     0.72101  |
| Llama-3.1-8B-Instruct     | 0.694433 |   0.702104 |     0.696528 |
| OpenHathi-7B-Hi-v0.1-Base | 0.556626 |   0.611409 |     0.605315 |
| Qwen2.5-7B-Instruct       | 0.625886 |   0.632473 |     0.604961 |
| aya-23-8B                 | 0.659441 |   0.650118 |     0.639472 |
| sarvam-1                  | 0.601778 |   0.601646 |     0.582922 |


## 📊 average_use_score

| Model                     |     Long |   Short_l6 |   Short_l6v2 |
|:--------------------------|---------:|-----------:|-------------:|
| Krutrim-2-instruct        | 0.767119 |   0.782181 |     0.763259 |
| Llama-3.1-8B-Instruct     | 0.765907 |   0.735025 |     0.710971 |
| OpenHathi-7B-Hi-v0.1-Base | 0.378686 |   0.466492 |     0.452251 |
| Qwen2.5-7B-Instruct       | 0.490086 |   0.48227  |     0.352178 |
| aya-23-8B                 | 0.661223 |   0.59273  |     0.552946 |
| sarvam-1                  | 0.558429 |   0.530562 |     0.465743 |


## 📊 average_labse_score

| Model                     |     Long |   Short_l6 |   Short_l6v2 |
|:--------------------------|---------:|-----------:|-------------:|
| Krutrim-2-instruct        | 0.753911 |   0.713546 |     0.684948 |
| Llama-3.1-8B-Instruct     | 0.728517 |   0.646643 |     0.614407 |
| OpenHathi-7B-Hi-v0.1-Base | 0.464447 |   0.443256 |     0.391588 |
| Qwen2.5-7B-Instruct       | 0.602324 |   0.557664 |     0.500421 |
| aya-23-8B                 | 0.575044 |   0.478557 |     0.417063 |
| sarvam-1                  | 0.505655 |   0.462654 |     0.421203 |


## 📊 average_laser_score

| Model                     |     Long |   Short_l6 |   Short_l6v2 |
|:--------------------------|---------:|-----------:|-------------:|
| Krutrim-2-instruct        | 0.79947  |   0.797246 |     0.796158 |
| Llama-3.1-8B-Instruct     | 0.815454 |   0.756917 |     0.749754 |
| OpenHathi-7B-Hi-v0.1-Base | 0.566163 |   0.573338 |     0.570828 |
| Qwen2.5-7B-Instruct       | 0.771087 |   0.752541 |     0.750139 |
| aya-23-8B                 | 0.758998 |   0.677248 |     0.659317 |
| sarvam-1                  | 0.703379 |   0.655234 |     0.638246 |


## 🏁 Overall Average (Mean of BERT, USE, LaBSE, LASER)

| Model                     |     Long |   Short_l6 |   Short_l6v2 |
|:--------------------------|---------:|-----------:|-------------:|
| Krutrim-2-instruct        | 0.765156 |   0.755676 |     0.741344 |
| Llama-3.1-8B-Instruct     | 0.751078 |   0.710172 |     0.692915 |
| OpenHathi-7B-Hi-v0.1-Base | 0.491481 |   0.523624 |     0.504995 |
| Qwen2.5-7B-Instruct       | 0.622346 |   0.606237 |     0.551925 |
| aya-23-8B                 | 0.663676 |   0.599663 |     0.5672   |
| sarvam-1                  | 0.59231  |   0.562524 |     0.527029 |

