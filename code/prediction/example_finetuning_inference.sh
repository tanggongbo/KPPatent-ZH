env CUDA_VISIBLE_DEVICES=0 env TRANSFORMERS_OFFLINE=1 llamafactory-cli train llama3_lora_sft_gemma3_4b.yaml

name=mozi3-7b
env CUDA_VISIBLE_DEVICES=0,1 env TRANSFORMERS_OFFLINE=1 python scripts/vllm_infer.py --model_name_or_path ../"$name"  --dataset keyphrase_patent_zh_test --template qwen --save_name ../keyphrase/prediction_"$name".jsonl  --temperature 0.0  --enable_thinking False --seed 100 >../keyphrase/log.infer."$name".base