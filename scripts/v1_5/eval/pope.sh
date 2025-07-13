#!/bin/bash
gpu_id=7
# model=liuhaotian/llava-v1.5-7b
# model=/home/yaolinli/code/LLaVA/official_ckpts/llava-v1.5-7b-lora
# model=/home/yaolinli/code/LLaVA/checkpoints/llava-v1.5-7b-lora
model=/home/yaolinli/code/LLaVA/checkpoints/llava-v1.5-7b-qformer-lora
output_name=llava-v1.5-7b-qformer-lora
llm=/home/shared/LargeLanguageModels/vicuna/vicuna-7b-v1.5

CUDA_VISIBLE_DEVICES=$gpu_id python -m llava.eval.model_vqa_loader \
    --model-path $model \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ./playground/data/eval/pope/val2014 \
    --answers-file ./playground/data/eval/pope/answers/$output_name.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --model-base $llm

python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/$output_name.jsonl >> /home/yaolinli/code/LLaVA/results/$output_name 2>&1
