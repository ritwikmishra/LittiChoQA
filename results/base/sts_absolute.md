# 🧾 STS Evaluation Results Summary

This report compares all models across evaluation metrics (BERT, USE, LaBSE, LASER).
---

## 📊 average_bert_score

| Model                     |     Long |   Short_l6 |   Short_l6v2 |
|:--------------------------|---------:|-----------:|-------------:|
| Krutrim-2-instruct        | 0.740126 |   0.729998 |     0.72125  |
| Llama-3.1-8B-Instruct     | 0.695364 |   0.702528 |     0.696753 |
| OpenHathi-7B-Hi-v0.1-Base | 0.555368 |   0.611765 |     0.605451 |
| Qwen2.5-7B-Instruct       | 0.625886 |   0.632402 |     0.604922 |
| aya-23-8B                 | 0.660201 |   0.650508 |     0.639623 |
| sarvam-1                  | 0.610365 |   0.601812 |     0.582064 |


## 📊 average_use_score

| Model                     |     Long |   Short_l6 |   Short_l6v2 |
|:--------------------------|---------:|-----------:|-------------:|
| Krutrim-2-instruct        | 0.767119 |   0.782595 |     0.763127 |
| Llama-3.1-8B-Instruct     | 0.767342 |   0.735192 |     0.711298 |
| OpenHathi-7B-Hi-v0.1-Base | 0.373972 |   0.465016 |     0.453539 |
| Qwen2.5-7B-Instruct       | 0.490086 |   0.482235 |     0.351742 |
| aya-23-8B                 | 0.664614 |   0.594487 |     0.553473 |
| sarvam-1                  | 0.548344 |   0.529103 |     0.46659  |


## 📊 average_labse_score

| Model                     |     Long |   Short_l6 |   Short_l6v2 |
|:--------------------------|---------:|-----------:|-------------:|
| Krutrim-2-instruct        | 0.753911 |   0.713419 |     0.684043 |
| Llama-3.1-8B-Instruct     | 0.727995 |   0.646774 |     0.614719 |
| OpenHathi-7B-Hi-v0.1-Base | 0.460927 |   0.443699 |     0.3931   |
| Qwen2.5-7B-Instruct       | 0.602324 |   0.557196 |     0.500714 |
| aya-23-8B                 | 0.577025 |   0.479377 |     0.419704 |
| sarvam-1                  | 0.518371 |   0.463274 |     0.420412 |


## 📊 average_laser_score

| Model                     |     Long |   Short_l6 |   Short_l6v2 |
|:--------------------------|---------:|-----------:|-------------:|
| Krutrim-2-instruct        | 0.799373 |   0.796642 |     0.796466 |
| Llama-3.1-8B-Instruct     | 0.818795 |   0.757109 |     0.750088 |
| OpenHathi-7B-Hi-v0.1-Base | 0.563859 |   0.575721 |     0.572027 |
| Qwen2.5-7B-Instruct       | 0.771085 |   0.753319 |     0.750301 |
| aya-23-8B                 | 0.761582 |   0.677441 |     0.660375 |
| sarvam-1                  | 0.714073 |   0.655442 |     0.637391 |


## 🏁 Overall Average (Mean of BERT, USE, LaBSE, LASER)

| Model                     |     Long |   Short_l6 |   Short_l6v2 |
|:--------------------------|---------:|-----------:|-------------:|
| Krutrim-2-instruct        | 0.765132 |   0.755663 |     0.741222 |
| Llama-3.1-8B-Instruct     | 0.752374 |   0.710401 |     0.693215 |
| OpenHathi-7B-Hi-v0.1-Base | 0.488532 |   0.52405  |     0.506029 |
| Qwen2.5-7B-Instruct       | 0.622345 |   0.606288 |     0.55192  |
| aya-23-8B                 | 0.665856 |   0.600453 |     0.568294 |
| sarvam-1                  | 0.597788 |   0.562408 |     0.526614 |

