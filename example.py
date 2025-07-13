from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
import pdb

model_path = "liuhaotian/llava-v1.5-7b"
# model_path = "/home/yaolinli/code/official_ckpts/LLaVA/llava-v1.5-7b-lora"


tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)


prompt = "What are the things I should be cautious about when I visit here?"
image_file = "https://llava-vl.github.io/static/images/view.jpg"

args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt,
    "conv_mode": None,
    "image_file": image_file,
    "sep": ",",
    "temperature": 1.0,
    "num_beams": 1,
    "max_new_tokens": 512,
    "top_p": 0.9,
    
})()


eval_model(args)