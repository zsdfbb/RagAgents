#!/bin/bash

.//home/zsdfbb/ssd_2t/ai_model/llama.cpp/main -m /home/zsdfbb/ssd_2t/ai_model/llama-2-7b/llama-2-7b.Q4_0.gguf --color \
    --ctx_size 2048 -n -1 -ins -b 256 --top_k 10000 \
    --temp 0.2 --repeat_penalty 1.1 -t 8 -ngl 10000

