import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import copy
from llava.explainability.llava_explain import *
from llava.constants import IMAGE_TOKEN_INDEX
import math
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from captum.attr import visualization
import requests
import pdb
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import nn
import sys

# from llava.explainability.refer import *  # Commented out as we're not using refcoco datasets
import time
from typing import List, Tuple
import pickle

def save_function_inputs(file_path, *args, **kwargs):
    with open(file_path, 'wb') as f:
        pickle.dump((args, kwargs), f)
    print("save function inputs to ", file_path)


def load_function_inputs(file_path):
    with open(file_path, 'rb') as f:
        args, kwargs = pickle.load(f)
    return args, kwargs

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False

set_seed(seed=42)

# === Bounding Box Utilities ===
def box_xywh2xyxy(bbox_xywh: List[float], img_wh: Tuple[int, int], do_normalize: bool = True) -> List[float]:
    bbox_xyxy = [bbox_xywh[0], bbox_xywh[1], bbox_xywh[0] + bbox_xywh[2], bbox_xywh[1] + bbox_xywh[3]]
    width, height = img_wh

    # Validate
    assert bbox_xyxy[0] < bbox_xyxy[2] <= width, "Invalid BBox Width!"
    assert bbox_xyxy[1] < bbox_xyxy[3] <= height, "Invalid BBox Height!"

    # Handle Normalization
    if do_normalize:
        bbox_xyxy = [bbox_xyxy[0] / width, bbox_xyxy[1] / height, bbox_xyxy[2] / width, bbox_xyxy[3] / height]

    # Return Box Coordinates rounded to 2 decimal places!
    return [round(coord, 2) for coord in bbox_xyxy]


def get_sgae_v(model, org_r_gaes, gt_bbx, image_file, visulize=False, target_cap=None, dataset=None, sname='GAE', apply_scale_area=True, last_word=False, max_word_score=False):
    # original images
    preprocess_img = model.preprocess_image(image_file)
    img_size = preprocess_img.shape[2]  # 336
    orig_image = Image.open(image_file)
    rgb_image = orig_image.convert('RGB')
    # convert image to a square image
    w, h = rgb_image.size   # (640, 516)
    a = img_size    # 336
    # convert image size (w,h) -> (a,a)
    # scale gt bbx list([x,y,w,h])
    img_size_wh = (w, h)
    normalized_box_xyxy = box_xywh2xyxy(gt_bbx, img_size_wh, do_normalize=True)
    scale_gt_bbx = [int(b * img_size) for b in normalized_box_xyxy]  # [10, 20, 7, 12] -> [8, 20, 5, 12]
    mask_query_interp = torch.zeros((img_size, img_size)).to(org_r_gaes[0].device)
    # set the value in gt bbx range to 1
    # finished -> TODO: double check the coordinate system
    # mask_query_interp[scale_gt_bbx[0]:scale_gt_bbx[2], scale_gt_bbx[1]:scale_gt_bbx[3]] = 1
    mask_query_interp[scale_gt_bbx[1]:scale_gt_bbx[3], scale_gt_bbx[0]:scale_gt_bbx[2]] = 1
    in_box_area = (normalized_box_xyxy[2] - normalized_box_xyxy[0]) * (normalized_box_xyxy[3] - normalized_box_xyxy[1])
    out_box_area = 1 - in_box_area
    if in_box_area <= 1e-6 or out_box_area <= 1e-6:
        # the gt_bbx annotation is error
        return False, None, None, None
    else:
        area_scale = in_box_area / out_box_area
    if math.isnan(area_scale):
        return False, None, None, None
    
    if max_word_score:
        max_s_gae_v, max_s_max, max_s_avg = list(), list(), list()
        max_s_gae_v_scale, max_s_max_scale, max_s_avg_scale = list(), list(), list()
    
    
    # S-GAE calculation refer to AMC-grounding/models/model_pretrain.py line 221
    # gt bbx: list([x,y,w,h])
    for org_r_gae in org_r_gaes:
        max_r_gae = org_r_gae
        
        r_gae = F.relu(org_r_gae, inplace=False)
        dim = int(r_gae.numel() ** 0.5)  # (24, 24)
        r_gae = r_gae.reshape(1, 1, dim, dim)   # [1, 1, 24, 24]

        # scale r_gae to scaled image size (336, 336)
        mask_size = img_size
        r_gae = F.interpolate(r_gae, size=mask_size, mode="bilinear", align_corners=False).squeeze()   # [336, 336]
        # normalize r_gae
        r_gae = normalize_rgae(r_gae, method='uniform_0_1')
        # scale r_gae.sum() = 1
        # r_gae = r_gae / r_gae.sum()
        
        in_box_values = r_gae * mask_query_interp
        out_box_values = r_gae * (1-mask_query_interp)
        
        # max
        max_in_box_values = in_box_values.max()
        max_out_box_values = out_box_values.max()
        s_max = max_in_box_values - max_out_box_values
        # assert int(in_box_values.sum() + out_box_values.sum()) == 1
        # average
        threshold = 0.5
        # filter out the values < threshold when calculate 
        in_box_values_flt = in_box_values[in_box_values > threshold]
        out_box_values_flt = out_box_values[out_box_values > threshold]
        if in_box_values_flt.numel() > 0:
            avg_in_box_values = torch.sum(in_box_values_flt)/in_box_values_flt.size(0)
        else:
            avg_in_box_values = torch.tensor([0.0]).to(out_box_values_flt.device)
        if out_box_values_flt.numel() > 0:
            avg_out_box_values = torch.sum(out_box_values_flt)/out_box_values_flt.size(0)
        else:
            avg_out_box_values = torch.tensor([0.0]).to(out_box_values_flt.device)
        # print("in_box_values: ", in_box_values[in_box_values > 0].sum(), "in_box_values_flt: ", in_box_values_flt.sum(), "out_box_values: ", out_box_values[out_box_values > 0].sum(), "\tout_box_values_flt: ", out_box_values_flt.sum())
        s_avg = avg_in_box_values - avg_out_box_values
        # calcualte S-GAE_v
        s_gae_v = (s_avg + s_max) / 2
        
        # get the value
        # scale from [-1, 1] to range [0, 100]
        s_gae_v, s_max, s_avg = s_gae_v.item(), s_max.item(), s_avg.item()
        if math.isnan(s_gae_v):
            print('s_gae_v is nan')
        
        s_gae_v, s_max, s_avg = (s_gae_v + 1) * 50, (s_max + 1) * 50, (s_avg + 1) * 50
        s_gae_v_scale, s_max_scale, s_avg_scale = s_gae_v/area_scale, s_max/area_scale, s_avg/area_scale
        
        if max_word_score:
            max_s_gae_v.append(s_gae_v), max_s_max.append(s_max), max_s_avg.append(s_avg)
            max_s_gae_v_scale.append(s_gae_v_scale), max_s_max_scale.append(s_max_scale), max_s_avg_scale.append(s_avg_scale)

    max_word_idx = -1
    if max_word_score:
        s_gae_v = max(max_s_gae_v)
        max_idx = max_s_gae_v.index(s_gae_v)
        s_max = max_s_max[max_idx]
        s_avg = max_s_avg[max_idx]
        max_word_idx = max_idx
        if apply_scale_area:
            s_gae_v_scale = max(max_s_gae_v_scale)
            max_idx_scale = max_s_gae_v_scale.index(s_gae_v_scale)
            s_max_scale = max_s_max_scale[max_idx_scale]
            s_avg_scale = max_s_avg_scale[max_idx_scale]
            max_word_idx = max_idx_scale
            
    print(f"S-{sname}_v: ", s_gae_v, "\tS_max: ", s_max, "\tS_avg: ", s_avg)
    if apply_scale_area:
        print(f"S-{sname}_v_scale: ", s_gae_v_scale, "\tS_max_scale: ", s_max_scale, "\tS_avg_scale: ", s_avg_scale, "\t area_scale: ", area_scale)
            
    if visulize:
        model_name = model.model_name
        scale_name = '_area-scaled' if apply_scale_area else ''
        if not os.path.exists(args.vis_save_path):
            os.mkdir(args.vis_save_path)
        save_path = os.path.join(args.vis_save_path, f"S-{sname}_{model_name}_v{scale_name}/")
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        vscores = [s_gae_v, s_max, s_avg] if not apply_scale_area else [s_gae_v_scale, s_max_scale, s_avg_scale]
        show_image_relevance(image_file, max_r_gae, preprocess_img, rtype='t2i', bbx=scale_gt_bbx, save_path=save_path, target_cap=target_cap, dataset=dataset, score=vscores, sname=sname, scale_area=apply_scale_area, last_word=last_word, max_word_score=max_word_score)
        
    if apply_scale_area:
        return True, s_gae_v_scale, s_max_scale, s_avg_scale, max_word_idx
    else:
        return True, s_gae_v, s_max, s_avg, max_word_idx
    

# load flickr30k-entities dataset
if __name__ == '__main__':
    # llava_model_path = '/home/yaolinli/code/LLaVA/official_ckpts/llava-v1.5-7b-lora'
    parser = argparse.ArgumentParser(description='Get S-GAE_v score for LLaVA model.')
    parser.add_argument('--llava_model_path', type=str, default='', help='LLaVA model paths')
    parser.add_argument('--mm_projector_type', type=str, default='', help='MM projector types')
    parser.add_argument('--model_name', type=str, default=None, help='model name when load from scratch')
    
    # Custom input parameters
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--target_text', type=str, required=True, help='Target text/phrase to analyze')
    parser.add_argument('--question', type=str, default='please describe the image in a sentence.', help='Question to ask about the image')
    parser.add_argument('--bbox', type=str, default=None, help='Bounding box in format "x,y,w,h" (optional, for S-GAE calculation with specific region)')
    
    parser.add_argument('--visualize', action='store_true', help='Whether to visulize the relevance map')
    parser.add_argument('--log_file', type=str, default='visualize_output/logs/q2i_analysis_log.txt', help='Log file path')
    parser.add_argument('--set_baseline', type=str, default=None, help='Baseline to compare with S-GAE_v, e.g. gradcam, rawattn or random.')
    parser.add_argument('--normalize_method', type=str, default=None, help='methods tonormalize the R-GAE_v: 1) uniform_0_1; 2) softmax; 3) row_uniform_0_1; 4) row_uniform_0_1_sum_1; 5) sum_1')
    parser.add_argument('--from_scratch', action='store_true', help='Whether to load the model from scratch.')
    parser.add_argument('--area_scale', action='store_true', help='Whether scale the final S-GAE score by bbx/img area ratio')
    parser.add_argument('--last_word', action='store_true', help='Whether only calculate the last word in the sentence.')
    parser.add_argument('--max_word_score', action='store_true', help='use the max score of the word in the sentence.')
    parser.add_argument('--min_txt_len', type=int, default=5, help='filter text length less than min_txt_len')
    parser.add_argument('--txt_pos_analysis', action='store_true', help='Whether to calculate each phrase in a caption to analyze the position bias. Please set the --last word in the meanwhile')
    parser.add_argument('--vis_save_path', type=str, default='visualize_output/', help='save path for q2i relevance map')
    parser.add_argument('--suffix', type=str, default='', help='suffix name added to the saved q2i maps for different analysis')
    # parser.add_argument('--q2i_only', action='store_true', help='only save the q2i map; default: q2i * i2i')
    
    args = parser.parse_args()
    if isinstance(args.llava_model_path, str):
        llava_model_paths = [args.llava_model_path]
    else:
        llava_model_paths = args.llava_model_path
    if isinstance(args.mm_projector_type, str):
        mm_projector_types = [args.mm_projector_type]
    else:
        mm_projector_types = args.mm_projector_type
    # llava_model_paths = ['/home/yaolinli/code/LLaVA/official_ckpts/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5', '/home/yaolinli/code/LLaVA/checkpoints/llava-v1.5-7b-qformer2_64-lora', '/home/yaolinli/code/LLaVA/checkpoints/llava-v1.5-7b-qformer2_64-pretrain']
    # mm_projector_types = ['mlp2x_gelu', 'qformer2_64', 'qformer2_64']
    assert len(llava_model_paths) == len(mm_projector_types)
    # llava_model_path = '/home/yaolinli/code/LLaVA/official_ckpts/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5' # '/home/yaolinli/code/LLaVA/official_ckpts/llava-v1.5-7b'
    # mm_projector_type = 'mlp2x_gelu'
    BASELINE = args.set_baseline  # None or 'gradcam' or 'rawattn'
    visualize = args.visualize
    log_file = args.log_file
    
    # Parse custom input parameters
    image_file = args.image_path
    target_text = args.target_text
    question = args.question
    
    # Parse bounding box if provided (format: "x,y,w,h")
    gt_bbx = None
    if args.bbox:
        try:
            bbox_parts = [float(x.strip()) for x in args.bbox.split(',')]
            if len(bbox_parts) == 4:
                gt_bbx = bbox_parts
                print(f"Using provided bounding box: {gt_bbx}")
            else:
                print("Warning: Invalid bounding box format. Expected 'x,y,w,h'. Proceeding without bounding box.")
        except ValueError:
            print("Warning: Invalid bounding box values. Proceeding without bounding box.")
    
    # Validate input files
    if not os.path.exists(image_file):
        print(f"Error: Image file {image_file} does not exist!")
        exit(1)
    
    # constant settings
    vicuna_model_path = 'lmsys/vicuna-7b-v1.5'
    mm_vision_tower_path = 'openai/clip-vit-large-patch14-336'
    pretrain_mm_qformer =  'Salesforce/instructblip-vicuna-7b' #'/home/yaolinli/code/llava_explain/ckpts/instructblip-7b-qformer' #
    pretrain_blip2_qformer = 'Salesforce/blip2-opt-2.7b'
    # vit settings from llava: related to attention layers
    mm_vision_select_layer = -2
    select_vit_feature = 'patch'
    version = 'v1'  # v0 means the training stage 1(image caption generation); v1 means the training stage 2 (instruction tuning)
    sname = 'rgae' if BASELINE is None else BASELINE
    
    for llava_model_path, mm_projector_type in zip(llava_model_paths, mm_projector_types):
        use_qformer = True
        if 'mlp' in mm_projector_type:
            use_qformer = False

        '''
        load llava model
        '''
        # lora
        print(llava_model_path, mm_projector_type, use_qformer)
        model_name = args.model_name if args.model_name is not None else llava_model_path
        if 'lora' in llava_model_path.lower() or 'pretrain' in llava_model_path.lower() or llava_model_path.lower() == '':
            model = ModelUsage(llava_model_path, vicuna_model_path, mm_vision_tower_path, mm_projector_type, pretrain_mm_qformer, conv_mode='vicuna_v1', version=version, device='cuda', model_name=model_name, from_scratch=args.from_scratch)
            # model = ModelUsage(llava_model_path, vicuna_model_path, mm_vision_tower_path, mm_projector_type, pretrain_mm_qformer, conv_mode='vicuna_v1', version=version, device='cpu', model_name=model_name, from_scratch=args.from_scratch)
        else:
        # full model
            model = ModelUsage(llava_model_path, None, mm_vision_tower_path, mm_projector_type, None, conv_mode='vicuna_v1', version=version, device='cuda', from_scratch=args.from_scratch)
        
        '''
        Process custom input
        '''
        print(f"Processing image: {image_file}")
        print(f"Target text: {target_text}")
        print(f"Question: {question}")
        
        # Process the target text
        cur_sent = target_text.replace("the ", "").replace("a ", "").replace("  ", "").strip().rstrip()
        cur_sent_len = len(cur_sent.split(" "))
        
        if cur_sent_len < args.min_txt_len:
            print(f"Warning: Target text is too short (less than {args.min_txt_len} words)")
        
        # Determine the text processing strategy
        if args.txt_pos_analysis:
            start_phrase_idx = 0
        else:
            start_phrase_idx = cur_sent_len - 1
            
        for si in range(start_phrase_idx, cur_sent_len):
            target_word = " ".join(cur_sent.split(" ")[:si+1]).rstrip()
            
            answer = f'{target_word} in the image.'

            '''
            Calculate GAE relevance map
            '''
            sent_input_ids = model.get_input_ids(question, answer)
            # torch.where(sent_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
            image_token_start_index = torch.where(sent_input_ids == IMAGE_TOKEN_INDEX)[1].tolist()[0]
            sent_input_ids = sent_input_ids[0].cpu().numpy().tolist()   # [1, 64]
            answer_input_ids = sent_input_ids[image_token_start_index:]
            word_input_ids = model.tokenizer.encode(target_word)[1:]    # [330, 3055, 17615]
            start_idx = find_indices(sent_input_ids, word_input_ids)[0] - 1  # left shift one position

            gen = Generator(model, select_vit_feature, mm_vision_select_layer)

            # get rel_map for R-GAE and other baselines e.g. gradcam and rawattn
            rel_scores = None
            if args.last_word:
                print("**only calcuate the last word in the sentence.**")
                print("target_word: ", target_word, "; raw sent: ", cur_sent)
            elif args.max_word_score:
                print("**only save the max word score among a sentence as th final S-GAE Score**")
                print("target_word: ", target_word, "; raw sent: ", cur_sent)
            
            if BASELINE is not None:
                gmap = get_baseline(gen, start_idx, word_input_ids, image_token_start_index,  image_file, question, answer, use_qformer, baseline=BASELINE, last_word=args.last_word, max_word_score=args.max_word_score, normalize_type='row_uniform_0_1_sum_1')
            else:
                gmap = get_rgae(gen, start_idx, word_input_ids, image_token_start_index, image_file, question, answer, use_qformer, last_word=args.last_word, max_word_score=args.max_word_score, normalize_type='row_uniform_0_1_sum_1')
            
                
                
            if use_qformer and torch.isnan(gmap['q2i']).any().item():
                print("q2i contains NaN:", torch.isnan(gmap['q2i']).any().item())
            if torch.isnan(gmap['i2i']).any().item():
                print("i2i contains NaN:", torch.isnan(gmap['i2i']).any().item())
            if torch.isnan(gmap['t2i']).any().item():
                print("t2i contains NaN:", torch.isnan(gmap['t2i']).any().item())
                
            
            norm_R_t_q, norm_R_t_i, norm_R_t_p, norm_R_q_i, norm_R_q_p, norm_R_i_p = None, None, None, None, None, None
            
            if use_qformer:
                # Q to Image tokens *after* vit encoder
                R_q_i = gmap['q2i'] 
                # Q to Image patches *before* vit encoder
                R_q_p = torch.matmul(gmap['q2i'], gmap['i2i']) 
                norm_R_q_i = normalize_rgae(R_q_i, method='row_uniform_0_1_sum_1')
                norm_R_q_p = normalize_rgae(R_q_p, method='row_uniform_0_1_sum_1')
                # LLM Text tokens T to query tokens Q 
                norm_R_t_q = normalize_rgae(gmap['t2q'].unsqueeze(0), method='row_uniform_0_1_sum_1').squeeze()
            
            # LLM Text tokens T to Image tokens *after* vit encoder
            R_t_i = torch.matmul(gmap['t2q'], gmap['q2i']) if use_qformer else gmap['t2q']
            norm_R_t_i = normalize_rgae(R_t_i.unsqueeze(0), method='row_uniform_0_1_sum_1').squeeze() # shape: ?
            
            # LLM Text tokens T to Image patches *before* vit encoder
            norm_R_t_p = normalize_rgae(gmap['t2i'].unsqueeze(0), method='row_uniform_0_1_sum_1').squeeze() # shape: [576]
            
            # Image tokens to Image patches *before* vit encoder
            norm_R_i_p_idx0 = normalize_rgae(gmap['i2i'][0].unsqueeze(0), method='row_uniform_0_1_sum_1').squeeze()
            norm_R_i_p_avg = normalize_rgae(gmap['i2i'].mean(axis=0).unsqueeze(0), method='row_uniform_0_1_sum_1').squeeze()

            
            preprocess_img = model.preprocess_image(image_file)
            
            target_word_str = '-'.join(target_word.split(" "))
            save_path = args.vis_save_path
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            
            
            save_dir = save_path + os.path.basename(image_file).split('.')[0]
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            rmap_type = args.set_baseline if args.set_baseline is not None else 'rgae'
            # save intermedia variables
            func_save_path = os.path.join(save_dir, f'{rmap_type}-{target_word_str}-{mm_projector_type}-{args.suffix}' + '.pkl')
            save_function_inputs(func_save_path, image_file, [norm_R_t_q, norm_R_t_i, norm_R_t_p, norm_R_q_i, norm_R_q_p, norm_R_i_p_avg, norm_R_i_p_idx0], ['R_t_q', 'R_t_i', 'R_t_p', 'R_q_i', 'R_q_p', 'R_i_p_avg', 'R_i_p_idx0'], preprocess_img, rtype=f'-{rmap_type}-{target_word_str}-{mm_projector_type}-{args.suffix}', img_size=336, save_path=save_path)

            loaded_args, loaded_kwargs = load_function_inputs(func_save_path)

            show_rel_maps(*loaded_args, **loaded_kwargs)
            
            # Optional: Calculate S-GAE_v score if bounding box is provided
            if gt_bbx is not None:
                # Calculate S-GAE_v score using the provided bounding box
                success, s_gae_v, s_max, s_avg, max_word_idx = get_sgae_v(
                    model, 
                    [norm_R_t_p], 
                    gt_bbx, 
                    image_file, 
                    visulize=visualize, 
                    target_cap=target_word, 
                    dataset='custom', 
                    sname=sname, 
                    apply_scale_area=args.area_scale, 
                    last_word=args.last_word, 
                    max_word_score=args.max_word_score
                )
                
                if success:
                    print(f"S-GAE_v score calculated successfully!")
                    print(f"S-GAE_v: {s_gae_v}, S_max: {s_max}, S_avg: {s_avg}")
                else:
                    print("Failed to calculate S-GAE_v score (invalid bounding box)")
            else:
                print("No bounding box provided, skipping S-GAE_v calculation")
                
        print(f"Processing completed for image: {image_file}")
        print(f"Results saved to: {save_path}")
                        
