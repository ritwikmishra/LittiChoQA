# 🧾 STS Evaluation Results Summary

This report compares all models across evaluation metrics (BERT, USE, LaBSE, LASER).
---

## 📊 average_bert_score

| Model                     |     Long |   Short_l6 |   Short_l6v2 |
|:--------------------------|---------:|-----------:|-------------:|
| Krutrim-2-instruct        | 0.744718 |   0.7406   |     0.718286 |
| Llama-3.1-8B-Instruct     | 0.699187 |   0.691234 |     0.673379 |
| OpenHathi-7B-Hi-v0.1-Base | 0.594914 |   0.543815 |     0.520952 |
| Qwen2.5-7B-Instruct       | 0.63038  |   0.621337 |     0.594025 |
| aya-23-8B                 | 0.667774 |   0.660912 |     0.632571 |
| sarvam-1                  | 0.606232 |   0.605891 |     0.565702 |


## 📊 average_use_score

| Model                     |     Long |   Short_l6 |   Short_l6v2 |
|:--------------------------|---------:|-----------:|-------------:|
| Krutrim-2-instruct        | 0.77527  |   0.753518 |     0.731653 |
| Llama-3.1-8B-Instruct     | 0.770457 |   0.76609  |     0.734467 |
| OpenHathi-7B-Hi-v0.1-Base | 0.410117 |   0.392284 |     0.338237 |
| Qwen2.5-7B-Instruct       | 0.482302 |   0.453479 |     0.39903  |
| aya-23-8B                 | 0.679548 |   0.655211 |     0.563    |
| sarvam-1                  | 0.550389 |   0.53077  |     0.439628 |


## 📊 average_labse_score

| Model                     |     Long |   Short_l6 |   Short_l6v2 |
|:--------------------------|---------:|-----------:|-------------:|
| Krutrim-2-instruct        | 0.747034 |   0.730781 |     0.666377 |
| Llama-3.1-8B-Instruct     | 0.724105 |   0.709748 |     0.65323  |
| OpenHathi-7B-Hi-v0.1-Base | 0.484607 |   0.472949 |     0.389385 |
| Qwen2.5-7B-Instruct       | 0.60452  |   0.584904 |     0.498413 |
| aya-23-8B                 | 0.591943 |   0.565964 |     0.468401 |
| sarvam-1                  | 0.51188  |   0.510827 |     0.369199 |


## 📊 average_laser_score

| Model                     |     Long |   Short_l6 |   Short_l6v2 |
|:--------------------------|---------:|-----------:|-------------:|
| Krutrim-2-instruct        | 0.778856 |   0.771276 |     0.739875 |
| Llama-3.1-8B-Instruct     | 0.794955 |   0.791868 |     0.773214 |
| OpenHathi-7B-Hi-v0.1-Base | 0.593741 |   0.542192 |     0.508309 |
| Qwen2.5-7B-Instruct       | 0.760301 |   0.756513 |     0.747131 |
| aya-23-8B                 | 0.743843 |   0.726572 |     0.659295 |
| sarvam-1                  | 0.686472 |   0.688525 |     0.597388 |


## 🏁 Overall Average (Mean of BERT, USE, LaBSE, LASER)

| Model                     |     Long |   Short_l6 |   Short_l6v2 |
|:--------------------------|---------:|-----------:|-------------:|
| Krutrim-2-instruct        | 0.76147  |   0.749044 |     0.714048 |
| Llama-3.1-8B-Instruct     | 0.747176 |   0.739735 |     0.708573 |
| OpenHathi-7B-Hi-v0.1-Base | 0.520845 |   0.48781  |     0.439221 |
| Qwen2.5-7B-Instruct       | 0.619376 |   0.604058 |     0.55965  |
| aya-23-8B                 | 0.670777 |   0.652165 |     0.580817 |
| sarvam-1                  | 0.588743 |   0.584004 |     0.492979 |

