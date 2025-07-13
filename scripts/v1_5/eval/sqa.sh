#!/bin/bash
gpu_id=6
# model=/home/yaolinli/code/LLaVA/official_ckpts/llava-v1.5-7b-lora
model=/home/yaolinli/code/LLaVA/checkpoints/llava-v1.5-7b-qformer-lora
output_name=llava-v1.5-7b-qformer-lora
llm=/home/shared/LargeLanguageModels/vicuna/vicuna-7b-v1.5

CUDA_VISIBLE_DEVICES=$gpu_id python -m llava.eval.model_vqa_science \
    --model-path $model \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/$output_name.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --model-base $llm

CUDA_VISIBLE_DEVICES=$gpu_id python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/$output_name'.jsonl' \
    --output-file ./playground/data/eval/scienceqa/answers/$output_name'_output.jsonl' \
    --output-result ./playground/data/eval/scienceqa/answers/$output_name'_result.json' >> /home/yaolinli/code/LLaVA/results/$output_name 2>&1
