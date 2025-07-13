#!/bin/bash
gpu_id=1
# model=liuhaotian/llava-v1.5-7b
# model=/home/yaolinli/code/LLaVA/official_ckpts/llava-v1.5-7b-lora
# model=/home/yaolinli/code/LLaVA/checkpoints/llava-v1.5-7b-lora
# output_name=llava-v1.5-7b-lora-reproduce
model=/home/yaolinli/code/LLaVA/checkpoints/llava-v1.5-7b-qformer-lora
output_name=llava-v1.5-7b-qformer-lora
llm=/home/shared/LargeLanguageModels/vicuna/vicuna-7b-v1.5


CUDA_VISIBLE_DEVICES=$gpu_id python -m llava.eval.model_vqa_loader \
    --model-path $model \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/$output_name.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --model-base $llm

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment $output_name

cd eval_tool

python calculation.py --results_dir answers/$output_name >> /home/yaolinli/code/LLaVA/results/$output_name 2>&1
