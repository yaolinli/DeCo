#!/bin/bash
gpu_id=2
model=/home/yaolinli/code/LLaVA/checkpoints/llava-v1.5-7b-lora
output_name=llava-v1.5-7b-lora-reproduce
# model=/home/yaolinli/code/LLaVA/official_ckpts/llava-v1.5-7b-lora
# output_name=llava-v1.5-7b-lora
# model=/home/yaolinli/code/LLaVA/checkpoints/llava-v1.5-7b-qformer-lora
# output_name=llava-v1.5-7b-qformer-lora
llm=/home/shared/LargeLanguageModels/vicuna/vicuna-7b-v1.5
key='7noixlklrymlnesh6wavwvwi790751c4q0yh0sp1lrd092r9'
chatgpt='gpt-4-1106-preview'

CUDA_VISIBLE_DEVICES=$gpu_id python -m llava.eval.model_vqa_mmhal \
    --model-path $model \
    --question-file ./playground/data/eval/MMHal-Bench/response_template.json \
    --image-folder ./playground/data/eval/MMHal-Bench/images \
    --answers-file ./playground/data/eval/MMHal-Bench/answers/$output_name.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --model-base $llm

# --response [JSON file with model responses] \
# --evaluation [JSON file with GPT-4 evaluation to be saved] \
# --api-key [your OpenAI API key, starting with 'sk-'] \
# --gpt-model [GPT model to be used, or 'gpt-4-0314' by default]

mkdir -p playground/data/eval/MMHal-Bench/reviews

python llava/eval/eval_gpt4_mmhal.py \
    --gpt-model $chatgpt \
    --api-key $key \
    --response ./playground/data/eval/MMHal-Bench/answers/$output_name.jsonl \
    --evaluation playground/data/eval/MMHal-Bench/reviews/$output_name.jsonl >> /home/yaolinli/code/LLaVA/results/$output_name 2>&1

