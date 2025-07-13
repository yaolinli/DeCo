# conda activate llava-explain

# if in China, please set:
export HF_ENDPOINT=https://hf-mirror.com

gpu_id=0

llava_model_paths=('liuhaotian/llava-v1.5-7b' 'yaolily/llava-v1.5-7b-qformer2_144-lora' 'yaolily/llava-v1.5-7b-avgpool2_144-lora')
mm_projector_types=('mlp2x_gelu' 'qformer2_144' 'avgpool2_144')

img='./examples/COCO_train2014_000000334463.jpg'
txt='A pot of green plants'
vis_save_path='./visualize_output/'
baseline='rgae' # default is rgae, others: 'rawattn' or 'gradcam'

length=${#llava_model_paths[@]}
for ((i=0; i<${length}; i++)); do
    llava_model_path=${llava_model_paths[$i]}
    mm_projector_type=${mm_projector_types[$i]}
    echo $llava_model_path
    echo $mm_projector_type

    CUDA_VISIBLE_DEVICES=$gpu_id python  ./llava/explainability/get_R-GAE_example.py --llava_model_path $model_path --mm_projector_type $mm_projector_type --visualize --vis_save_path $vis_save_path --image_path $img  --target_text $txt --set_baseline $baseline
done
