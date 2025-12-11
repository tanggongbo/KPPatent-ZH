#!/bin/bash

name="mozi3-7b"

for task in "scq" "mcq" "tf"
do
    for group in "subword" "semantic"
    do 
        dataset_name="keyphrase_patent_zh_${task}_${group}"
        env CUDA_VISIBLE_DEVICES=0 env TRANSFORMERS_OFFLINE=1 \
        python scripts/vllm_infer.py \
        --model_name_or_path ../../${name} \
        --dataset ${dataset_name} \
        --template qwen \
        --cutoff_len 4000 \
        --temperature 0.0  \
        --enable_thinking False \
        --save_name ../../keyphrase/prediction_test_${name}_${task}_${group}.jsonl  \
        --seed 100 >> ../../keyphrase/log.infer.${name}.additional

        echo "finished keyphrase_patent_zh_${task}_${group}"
    done
    
done

