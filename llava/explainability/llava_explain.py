import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import copy
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.train.train import preprocess_multimodal, preprocess_v1, preprocess_plain
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from transformers import InstructBlipProcessor
import math
from llava.explainability.ExplanationGenerator import Generator
import random
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from captum.attr import visualization
import requests
import pdb
import torch.nn.functional as F


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

def expand2square(pil_img, background_color):
    # this pads the non-square images to square, instead of cropping them; it slightly reduces hallucination.
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result                


def normalize_rgae(r_gae, method='row_uniform_0_1_sum_1'):
    r_gae = F.relu(r_gae, inplace=False)  # make sure value > 0 in the r_gae
    # normalize r_gae
    h, w = r_gae.shape
    device = r_gae.device
    dtype = r_gae.dtype
    if method == 'uniform_0_1':
        if (r_gae.max() - r_gae.min()) < 1e-12:  # The denominator is zero
            r_gae = torch.ones(h, w).to(dtype).to(device) / w*h
        else:
            # The denominator is not zero
            r_gae = (r_gae - r_gae.min()) / (r_gae.max() - r_gae.min())  
    elif method == 'softmax':
    # normalize each row of r_gae to sum to 1
        r_gae = F.softmax(r_gae, dim=1)
    elif method == 'row_uniform_0_1':
    # scale the value of ecah row in r_gae to [0, 1]
        for i in range(r_gae.shape[0]):  # The denominator is zero
            if ((r_gae[i].max() - r_gae[i].min()) < 1e-12):
                r_gae[i] = torch.ones(w).to(dtype).to(device) / w
            else:
                r_gae[i] = (r_gae[i] - r_gae[i].min()) / (r_gae[i].max() - r_gae[i].min())
    elif method == 'row_uniform_0_1_sum_1':
    # scale the value of ecah row in r_gae to [0, 1]
        for i in range(r_gae.shape[0]):
            if ((r_gae[i].max() - r_gae[i].min()) < 1e-12):
                r_gae[i] = torch.ones(w).to(dtype).to(device) / w
            else:
                r_gae[i] = (r_gae[i] - r_gae[i].min()) / (r_gae[i].max() - r_gae[i].min())
            r_gae[i] = r_gae[i] / (r_gae[i].sum())
    elif method == 'sum_1':
    # scale r_gae.sum() = 1
        if (r_gae.sum() < 1e-12):
            r_gae = torch.ones(h, w).to(dtype).to(device) / w*h
        else:
            r_gae = r_gae / (r_gae.sum())
    elif method == 'row_sum_1':
    # scale r_gae.sum() = 1
        for i in range(r_gae.shape[0]):
            if (r_gae.sum() < 1e-12):
                r_gae[i] = torch.ones(w).to(dtype).to(device) / w
            else:
                r_gae[i] = r_gae[i] / (r_gae[i].sum())
    elif method is None:
        return r_gae
    else:
        raise ValueError('method not supported.')
    return r_gae

                
class ModelUsage:
    def __init__(self, model_path, model_base=None, mm_vision_tower=None, mm_projector_type='mlp2x_gelu', pretrain_mm_qformer='Salesforce/instructblip-vicuna-7b', conv_mode='vicuna_v1', version='v1', device='cuda', model_name=None, from_scratch=False):
        disable_torch_init()
        if model_name is None:
            model_name = model_path
        self.model_name = get_model_name_from_path(model_name)
        self.model_path = os.path.expanduser(model_path)
        self.model_base = model_base
        print('self.model_name', self.model_name)
        print('self.model_path', self.model_path)
        # load models and model components
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(self.model_path, self.model_base, self.model_name, from_scratch=from_scratch)
        self.mm_projector_type = mm_projector_type
        self.pretrain_mm_qformer = pretrain_mm_qformer
        self.conv_mode = conv_mode
        self.version = version
        if 'mmqformer' in self.mm_projector_type:
            self.qformer_tokenizer = InstructBlipProcessor.from_pretrained(self.pretrain_mm_qformer).qformer_tokenizer
        self.clip_vit = self.model.get_vision_tower()
        self.image_boxes_len = self.clip_vit.num_patches  # (336/14)^2 = 576
        
        self.device = device
        self.model.train()
        self.model.to(self.device)
    

    def forward(self, image_file, question, answer, q_instruction=None):
        self.image_file_path = image_file
        # get image features
        image = Image.open(image_file).convert('RGB')
        # convert image to a square image
        w, h = image.size
        a = min(w, h)
        image_resized = image.resize((a, a))
        # this pads the non-square images to square, instead of cropping them; it slightly reduces hallucination.
        image_tensor = self.image_processor.preprocess(image_resized, return_tensors='pt')['pixel_values'][0]
        images = image_tensor.unsqueeze(0).half().to(self.device)
        
        # preprocess input text
        sources = [{'conversations': [{'from': 'human', 'value': '<image>\n'+question}, {'from': 'gpt', 'value': answer}]}]
        self.model.config.is_multimodal = True
        sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.model.config)
        
        if self.version == 'v1':  # training stage 2: instruction tuning
            data_dict = preprocess_v1(sources, self.tokenizer, has_image=True)
        else:  # training stage 1
            data_dict = preprocess_plain(sources, self.tokenizer)
        
        input_ids = data_dict['input_ids'][0].unsqueeze(0).to(self.device)
        labels = data_dict['labels'][0].unsqueeze(0).to(self.device)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        if 'mmqformer' in self.mm_projector_type:
            q_instruction = question.strip()
            assert q_instruction is not None, 'q_instruction is required for mmqformer'
            qformer_input_ids = self.qformer_tokenizer(q_instruction, truncation=True, return_tensors='pt')['input_ids'].to(self.device)
            
            output_dicts = self.model(
                input_ids=input_ids,
                qformer_input_ids=qformer_input_ids,    # add qformer input ids for instruction-aware qformer struction
                labels=labels,
                images=images,
                attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
                output_attentions=True
            )
        else:
            output_dicts = self.model(
                input_ids=input_ids,
                images=images,
                labels=labels,
                attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
                output_attentions=True
            )
        self.outputs = output_dicts
        return self.outputs, input_ids

    def preprocess_image(self, image_file):
        image = Image.open(image_file).convert('RGB')
        # convert image to a square image
        w, h = image.size
        a = min(w, h)
        image_resized = image.resize((a, a))
        # this pads the non-square images to square, instead of cropping them; it slightly reduces hallucination.
        image_tensor = self.image_processor.preprocess(image_resized, return_tensors='pt')['pixel_values'][0]
        images = image_tensor.unsqueeze(0).half().to(self.device)
        return images
    
    def get_input_ids(self, question, answer):
        # preprocess input text
        sources = [{'conversations': [{'from': 'human', 'value': '<image>\n'+question}, {'from': 'gpt', 'value': answer}]}]
        self.model.config.is_multimodal = True
        sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.model.config)
        
        if self.version == 'v1':  # training stage 2: instruction tuning
            data_dict = preprocess_v1(sources, self.tokenizer, has_image=True)
        else:  # training stage 1
            data_dict = preprocess_plain(sources, self.tokenizer)
        
        input_ids = data_dict['input_ids'][0].unsqueeze(0).to(self.device)
        labels = data_dict['labels'][0].unsqueeze(0).to(self.device)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        return input_ids
    
    def __call__(self, image_file, question, answer, q_instruction=None):
        return self.forward(image_file, question, answer, q_instruction)


def find_indices(sent_list, word_list):
    indices = []
    len_word_list = len(word_list)

    for i in range(len(sent_list) - len_word_list + 1):
        # Check if the slice of sent_list at this position matches word_list
        if sent_list[i:i + len_word_list] == word_list:
            indices.append(i)

    return indices

def save_image_vis(image_file_path, bbox_scores):
    bbox_scores = rel_map
    _, top_bboxes_indices = bbox_scores.topk(k=1, dim=-1)
    img = cv2.imread(image_file_path)
    mask = torch.zeros(img.shape[0], img.shape[1])
    for index in range(len(bbox_scores)):
        [x, y, w, h] = model_lrp.bboxes[0][index]
        curr_score_tensor = mask[int(y):int(h), int(x):int(w)]
        new_score_tensor = torch.ones_like(curr_score_tensor)*bbox_scores[index].item()
        mask[int(y):int(h), int(x):int(w)] = torch.max(new_score_tensor,mask[int(y):int(h), int(x):int(w)])
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    mask = mask.unsqueeze_(-1)
    mask = mask.expand(img.shape)
    img = img * mask.cpu().data.numpy()
    save_image_name = os.path.dirname(image_file_path) + '/vis_' + os.path.basename(image_file_path)
    cv2.imwrite(save_image_name, img)
    print("save ", save_image_name)


def show_image_relevance(image_path, image_relevance, image, rtype='t2i', query_idx=None, img_size=336, bbx=None, save_path=None, target_cap='', dataset='', score=None, sname='GAE', scale_area=False, last_word=False, max_word_score=False):
    # create heatmap from mask on image
    scale_name = '_area-scaled' if scale_area else ''
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam

    img_size = image.shape[2]
    orig_image = Image.open(image_path)
    rgb_image = orig_image.convert('RGB')
    # convert image to a square image
    w, h = rgb_image.size
    a = min(w, h)
    orig_image = rgb_image.resize((a, a))
        
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(orig_image)
    axs[0].axis('off')

    dim = int(image_relevance.numel() ** 0.5)  # (24, 24)
    image_relevance = image_relevance.reshape(1, 1, dim, dim)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=img_size, mode='bilinear')
    image_relevance = image_relevance.reshape(img_size, img_size).cuda().data.cpu().numpy()
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    image = image[0].permute(1, 2, 0).data.cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())
    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    axs[1].imshow(vis)
    if bbx is not None:
        xmin, ymin, xmax, ymax = bbx
        axs[1].add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, fill=False, color='red', linewidth=3))
    if target_cap != '':
        # write the word to the pic as the title
        axs[0].set_title(target_cap)
    if score is not None:
        # write the score as the text on the bottom
        xlim = axs[0].get_xlim()
        ylim = axs[0].get_ylim()
        text_x = xlim[0] + (xlim[1] * 0.2)  # leave 10% space
        text_y = ylim[0] + (ylim[0] * 0.1)  # leave 2% space
        axs[0].text(text_x, text_y, f'S-{sname}_v{scale_name}: {score[0]:.2f}; S_max: {score[1]:.2f}; S_avg: {score[2]:.2f}', fontsize=12, color='white', backgroundcolor='black', horizontalalignment='left')
    axs[1].axis('off')
    
    if save_path is None:
        save_path = os.path.dirname(image_path)
    
    last_word_str = ''
    if last_word:
        last_word_str = '_last-word'
    elif max_word_score:
        last_word_str = '_max-word'

    if query_idx is None:
        save_image_name = save_path + f'/{dataset}_{rtype}_' + os.path.basename(image_path).split('.')[0] + last_word_str + f'_{target_cap}' + '.jpg'
    else:
        save_image_name = save_path + f'/{dataset}_{rtype}_' + os.path.basename(image_path).split('.')[0] + '_query_' + str(query_idx) + last_word_str + f'_{target_cap}' + '.jpg'
    fig.savefig(save_image_name, bbox_inches='tight')
    print("save ", save_image_name)
    return


def get_rgae(gen, start_idx, word_input_ids, image_token_start_index, image_file, question, answer, use_qformer=False, last_word=False, max_word_score=False, normalize_type='row_uniform_0_1_sum_1'):
    rel_map = {}
    # Solving the participle problem by aagregating the relevance map of each sub-word 
    # # e.g. 'giraffe' -> 'g ira ffe'
    for i in range(len(word_input_ids)):
        if last_word and i < (len(word_input_ids) - 1):  # only calculate the last word in the text caption:
            continue
        cur_start_idx = start_idx + i
        # print("===================================")
        # print("subword index: {}\tstart_idx:{}\tword_input_id:{}".format(i, cur_start_idx, word_input_ids[i]))
        
        # target_index: the indicated word index in the the raw cap
        # index: predict word index
        # q means query token, i means visual tokens, qt means llm input tokens
        # self.R_qt_qt.shape: [639, 639]
        # self.R_i_i.shape: [576, 576]
        # self.R_q_i: [64, 576]
        R_qt_qt, R_i_i, R_q_i = gen.generate_ours(image_file, question, answer, target_index=cur_start_idx, index=word_input_ids[i], normalize_self_attention=True)
        
        # R_qt_qt = normalize_rgae(R_qt_qt, method=normalize_type)
        # R_i_i = normalize_rgae(R_i_i, method=normalize_type)
        # if use_qformer:
        #     R_q_i = normalize_rgae(R_q_i, method=normalize_type)
        # get Relevance map R_a_q of answer text tokens to visual tokens
        ## The qt_qt map is [t1, q, t2] * [t1, q, t2] beacuse it inserts <image> in the input text tokens
        qnum = R_q_i.shape[0] if use_qformer else R_i_i.shape[0]
        R_a_q = R_qt_qt[(image_token_start_index + qnum):, image_token_start_index:(image_token_start_index + qnum)]

        # answer text tokens to question text tokens
        R_a_qs = R_qt_qt[(image_token_start_index + qnum):, :image_token_start_index]
        R_a_a = R_qt_qt[(image_token_start_index + qnum):, (image_token_start_index + qnum):]
        # print("R_answer_vision.sum: {};\tR_answer_question.sum: {};\tR_answer_answer.sum: {}".format(R_a_q.sum(), R_a_qs.sum(), R_a_a.sum()))
        
        # visualize the relevance map
        # LLM tokens to image tokens
        if use_qformer:
            R_t_i = torch.matmul(torch.matmul(R_a_q, R_q_i), R_i_i)  # torch.Size([27, 576])
            R_q_i_aggr = torch.matmul(R_q_i, R_i_i)  # torch.Size([64, 576])
        else:
            # in this case: q = i
            R_t_i = torch.matmul(R_a_q, R_i_i)  # torch.Size([27, 576])
        # get the target word (e.g. 'giraffe') idx
        start_idx_ans = cur_start_idx - image_token_start_index - 1
        if 't2i' not in rel_map:
            rel_map['t2i'] = [R_t_i[start_idx_ans]]  # torch.Size([576])
        else:
            rel_map['t2i'].append(R_t_i[start_idx_ans])
            
        if 't2q' not in rel_map:
            rel_map['t2q'] = [R_a_q[start_idx_ans]]  # torch.Size([576])
        else:
            rel_map['t2q'].append(R_a_q[start_idx_ans])
            
        if use_qformer:
            if 'q2i' not in rel_map:
                rel_map['q2i'] = [R_q_i]  # torch.Size([576])
            else:
                rel_map['q2i'].append(R_q_i)
        
        if 'i2i' not in rel_map:
            rel_map['i2i'] = [R_i_i]
        else:
            rel_map['i2i'].append(R_i_i)
            
    # average the relevance map of each sub-word
    if not max_word_score:
        for key in rel_map:
            rel_map[key] = sum(rel_map[key]) / len(word_input_ids)
    return rel_map


def get_baseline(gen, start_idx, word_input_ids, image_token_start_index, image_file, question, answer, use_qformer=False, baseline='gradcam', last_word=False, max_word_score=False, normalize_type=None):
    rel_map = {}
    # Solving the participle problem by aagregating the relevance map of each sub-word 
    # # e.g. 'giraffe' -> 'g ira ffe'
    for i in range(len(word_input_ids)):
        if last_word and i < (len(word_input_ids) - 1):  # only calculate the last word in the text caption
            continue
        cur_start_idx = start_idx + i
        # print("===================================")
        # print("subword index: {}\tstart_idx:{}\tword_input_id:{}".format(i, cur_start_idx, word_input_ids[i]))
        
        # target_index: the indicated word index in the the raw cap
        # index: predict word index
        # q means query token, i means visual tokens, qt means llm input tokens

        # get rel_map
        if baseline == 'gradcam':
            # G_qt_qt: [635, 635]
            # G_q_i: [64, 576]
            # G_i_i: [576, 576]
            G_qt_qt, G_q_i, G_i_i = gen.generate_attn_gradcam(image_file, question, answer, target_index=cur_start_idx, index=word_input_ids[i])
        elif baseline == 'rawattn':
            G_qt_qt, G_q_i, G_i_i = gen.generate_raw_attn(image_file, question, answer, target_index=cur_start_idx, index=word_input_ids[i])
        elif baseline == 'random':
            G_qt_qt_refer, G_q_i_refer, G_i_i_refer = gen.generate_raw_attn(image_file, question, answer, target_index=cur_start_idx, index=word_input_ids[i])
            G_qt_qt = torch.randn_like(G_qt_qt_refer)
            G_i_i = torch.randn_like(G_i_i_refer)
            G_i_i = normalize_rgae(G_i_i, method=normalize_type)
            G_qt_qt = normalize_rgae(G_qt_qt, method=normalize_type)
            if use_qformer:
                G_q_i = torch.randn_like(G_q_i_refer)
                G_q_i = normalize_rgae(G_q_i, method=normalize_type)
        else:
            raise ValueError('baseline should be gradcam or rawattn')
        # get Relevance map R_a_q of answer text tokens to visual tokens
        ## The qt_qt map is [t1, q, t2] * [t1, q, t2] beacuse it inserts <image> in the input text tokens
        
        # G_qt_qt = normalize_rgae(G_qt_qt, method=normalize_type)
        # G_i_i = normalize_rgae(G_i_i, method=normalize_type)
        # print("normalized G_i_i contains NaN:", torch.isnan(G_i_i).any().item())
        # if torch.isnan(G_i_i).any().item():
        #     print("normalized G_i_i contains NaN:", G_i_i)
        
        # if use_qformer:
        #     G_q_i = normalize_rgae(G_q_i, method=normalize_type)
            
        qnum = G_q_i.shape[0] if use_qformer else G_i_i.shape[-1]
        G_a_q = G_qt_qt[(image_token_start_index + qnum):, image_token_start_index:(image_token_start_index + qnum)]
        # answer text tokens to question text tokens
        # print("G_answer_vision.sum: {}".format(G_a_q.sum()))
        
        # visualize the relevance map
        # LLM tokens to image tokens
        if use_qformer:
            G_t_i = torch.matmul(G_a_q, G_q_i)  # torch.Size([27, 576])
        else:
            # in this case: q = i
            G_t_i = G_a_q  # torch.Size([27, 576])
        # get the target word (e.g. 'giraffe') idx
        start_idx_ans = cur_start_idx - image_token_start_index - 1
        if 't2i' not in rel_map:
            rel_map['t2i'] = [G_t_i[start_idx_ans]]
        else:
            rel_map['t2i'].append(G_t_i[start_idx_ans])
        
        if 't2q' not in rel_map:
            rel_map['t2q'] = [G_a_q[start_idx_ans]]  # torch.Size([576])
        else:
            rel_map['t2q'].append(G_a_q[start_idx_ans])
            
        if use_qformer:   
            if 'q2i' not in rel_map:
                rel_map['q2i'] = [G_q_i]
            else:
                rel_map['q2i'].append(G_q_i)
        
        if 'i2i' not in rel_map:
            rel_map['i2i'] = [G_i_i]
        else:
            rel_map['i2i'].append(G_i_i)

    # average the relevance map of each sub-word
    if not max_word_score:
        for key in rel_map:
            rel_map[key] = sum(rel_map[key]) / len(word_input_ids)
    return rel_map






def show_rel_maps(image_path, rmaps, rnames, preprocess_image, rtype='t2i', img_size=336, save_path='/home/yaolinli/code/LLaVA/llava/explainability/examples/q2i_visualize/', white_preprocess_img=None, save_img_type='.png'):
    # create heatmap from mask on image
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam

    
    save_dir = save_path + os.path.basename(image_path).split('.')[0]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    
    for i, image_relevance in enumerate(rmaps):
        rname = rnames[i]
        img_size = preprocess_image.shape[2]
        orig_image = Image.open(image_path)
        rgb_image = orig_image.convert('RGB')
        # convert image to a square image
        w, h = rgb_image.size
        a = min(w, h)
        orig_image = rgb_image.resize((a, a))


        if image_relevance is None:  # mlp2x_gelu does not have R_q_i & R_q_p & R_t_q
            if rnames[i] == 'R_q_i':
                image_relevance = torch.ones(img_size, img_size).cuda().to(rmaps[1].dtype) / (img_size * img_size)
            elif rnames[i] == 'R_t_q':  # 'R_t_q' == 'R_t_i' when projector is mlp2x_gelu
                image_relevance = rmaps[1]  
                
            else:
                continue
        
        
        save_image_name = os.path.join(save_dir, f'{rname}{rtype}' + save_img_type) 
        # if rname in ['R_t_i', 'R_t_p', 'R_i_p', 'R_i_p_avg', 'R_i_p_idx0']:
        if rname in ['R_t_i']:
            fig, axs = plt.subplots(1, 1)
            # axs[0].imshow(orig_image)
            # axs[0].axis('off')

            dim = int(image_relevance.numel() ** 0.5)  # (24, 24)
            image_relevance = image_relevance.reshape(1, 1, dim, dim)
            image_relevance = torch.nn.functional.interpolate(image_relevance, size=img_size, mode='bilinear')
            image_relevance = image_relevance.reshape(img_size, img_size).cuda().data.cpu().numpy()
            image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
            image = preprocess_image[0].permute(1, 2, 0).data.cpu().numpy()
            image = (image - image.min()) / (image.max() - image.min())
            vis = show_cam_on_image(image, image_relevance)
            vis = np.uint8(255 * vis)
            vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
            axs.imshow(vis)
            axs.axis('off')
            fig.savefig(save_image_name, bbox_inches='tight')
            fig.savefig(save_image_name.replace(save_img_type, '.svg'), bbox_inches='tight')
        elif rname in ['R_t_q']:
            '''
            Original method: white img + entire t -> q            
            '''
            dim = int(image_relevance.numel() ** 0.5)  # (14, 14)
            fig, axs = plt.subplots(1, 1)

            image_relevance_normalized = image_relevance.reshape(1, 1, dim, dim)
            image_relevance_normalized = torch.nn.functional.interpolate(image_relevance_normalized, size=img_size, mode='bilinear')
            image_relevance_normalized = image_relevance_normalized.reshape(img_size, img_size).cuda().data.cpu().numpy()
            image_relevance_normalized = (image_relevance_normalized - image_relevance_normalized.min()) / (image_relevance_normalized.max() - image_relevance_normalized.min())
            
            alpha = 0.5 # Adjust this multiplier to control transparency
            white_image = np.ones((img_size, img_size, 3)) * alpha
            heatmap = cv2.applyColorMap(np.uint8(255 * image_relevance_normalized), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            cam = heatmap * 0.5  + np.float32(white_image) * 0.5
            vis = cam
            vis = np.uint8(255 * vis)
            vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
            axs.imshow(vis)
            axs.axis('off')
            
            # Save original version
            fig.savefig(save_image_name, bbox_inches='tight')
            fig.savefig(save_image_name.replace(save_img_type, '.svg'), bbox_inches='tight')
            plt.close(fig)
            
            '''
            Additional Grid-based visualization for R_t_q to match R_q_i style
            '''
            # Create grid layout similar to R_q_i
            n = int(image_relevance.numel() ** 0.5)  # Calculate grid size (e.g., 8x8 for 64 elements)
            fig_grid, axs_grid = plt.subplots(n, n, figsize=(n*2, n*2))
            plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.1, hspace=0.1)
            
            # Normalize the relevance values
            image_relevance_norm = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
            image_relevance_cpu = image_relevance_norm.data.cpu().numpy()

            for query_idx in range(n * n):
                xi = query_idx // n
                yi = query_idx % n
                
                # Get the relevance score for this query token
                if query_idx < len(image_relevance_cpu):
                    relevance_score = image_relevance_cpu[query_idx]
                else:
                    relevance_score = 0.0
                
                # Create a uniform colored image based on the relevance score
                alpha = 0.3  # Background transparency
                white_image = np.ones((img_size, img_size, 3)) * alpha
                
                # Create a uniform color map for this grid cell
                uniform_relevance = np.ones((img_size, img_size)) * relevance_score
                heatmap = cv2.applyColorMap(np.uint8(255 * uniform_relevance), cv2.COLORMAP_JET)
                heatmap = np.float32(heatmap) / 255
                
                # Blend with white background
                vis = 0.7 * heatmap + 0.3 * white_image
                vis = np.uint8(255 * vis)
                vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
                
                axs_grid[xi][yi].imshow(vis)
                axs_grid[xi][yi].axis('off')
                
                # Optionally add text showing the relevance score
                axs_grid[xi][yi].text(img_size//2, img_size//2, f'{relevance_score:.2f}', 
                                   ha='center', va='center', fontsize=8, color='white', 
                                   bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
            
            # Save grid version with "grid" in filename
            grid_save_name = save_image_name.replace(save_img_type, f'_grid{save_img_type}')
            fig_grid.savefig(grid_save_name, bbox_inches='tight')
            fig_grid.savefig(grid_save_name.replace(save_img_type, '.svg'), bbox_inches='tight')
            plt.close(fig_grid)
            
        # elif rname in ['R_q_i', 'R_q_p']:
        elif rname in ['R_q_i']:
            if 'mlp2x_gelu' in rtype:  
                # mlp2x_gelu does not have R_q_i 
                # specially processing: draw the original image 
                fig, axs = plt.subplots(1, 1)
                image_relevance = image_relevance.data.cpu().numpy()
                image = preprocess_image[0].permute(1, 2, 0).data.cpu().numpy()
                image = (image - image.min()) / (image.max() - image.min())
                heatmap = cv2.applyColorMap(np.uint8(255 * image_relevance), cv2.COLORMAP_JET)
                heatmap = np.float32(heatmap) / 255
                cam = heatmap*0.5  + np.float32(image)*0.5
                # vis = cam / np.max(cam)
                vis = cam
                
                vis = np.uint8(255 * vis)
                vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
                axs.imshow(vis)
                axs.axis('off')
                
            else:
                n = int(image_relevance.shape[0] ** 0.5)
                fig, axs = plt.subplots(n, n, figsize=(n*2, n*2))
                plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.1, hspace=0.1)
                # axs[0][0].imshow(orig_image)
                # axs[0][0].axis('off')
                
                for query_idx in range(image_relevance.shape[0]):
                    # xi = (query_idx+1) // n
                    # yi = (query_idx+1) % n
                    xi = (query_idx) // n
                    yi = (query_idx) % n
                    dim = int(image_relevance[query_idx].numel() ** 0.5)  # (24, 24)
                    cur_image_relevance = image_relevance[query_idx].reshape(1, 1, dim, dim)
                    cur_image_relevance = torch.nn.functional.interpolate(cur_image_relevance, size=img_size, mode='bilinear')
                    cur_image_relevance = cur_image_relevance.reshape(img_size, img_size).cuda().data.cpu().numpy()
                    cur_image_relevance = (cur_image_relevance - cur_image_relevance.min()) / (cur_image_relevance.max() - cur_image_relevance.min())
                    image = preprocess_image[0].permute(1, 2, 0).data.cpu().numpy()
                    image = (image - image.min()) / (image.max() - image.min())
                    vis = show_cam_on_image(image, cur_image_relevance)
                    vis = np.uint8(255 * vis)
                    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
                    axs[xi][yi].imshow(vis)
                    axs[xi][yi].axis('off')
            fig.savefig(save_image_name, dpi=300, bbox_inches='tight')
            fig.savefig(save_image_name.replace(save_img_type, '.svg'), bbox_inches='tight')
        print("save at: ", save_image_name)
    return


def show_query_heads(image_path, image_relevance, preprocess_image, rtype='t2i', img_size=336, save_path='/home/yaolinli/code/LLaVA/llava/explainability/examples/q2i_visualize/'):
    # create heatmap from mask on image
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam

    img_size = preprocess_image.shape[2]
    orig_image = Image.open(image_path)
    rgb_image = orig_image.convert('RGB')
    # convert image to a square image
    w, h = rgb_image.size
    a = min(w, h)
    orig_image = rgb_image.resize((a, a))
    
    n = int(image_relevance.shape[0] ** 0.5)
    fig, axs = plt.subplots(n, n, figsize=(n*2, n*2))
    
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.1, hspace=0.1)
    axs[0][0].imshow(orig_image)
    axs[0][0].axis('off')
    
    for query_idx in range(image_relevance.shape[0]-1):
        xi = (query_idx+1) // n
        yi = (query_idx+1) % n
        dim = int(image_relevance[query_idx].numel() ** 0.5)  # (24, 24)
        cur_image_relevance = image_relevance[query_idx].reshape(1, 1, dim, dim)
        cur_image_relevance = torch.nn.functional.interpolate(cur_image_relevance, size=img_size, mode='bilinear')
        cur_image_relevance = cur_image_relevance.reshape(img_size, img_size).cuda().data.cpu().numpy()
        cur_image_relevance = (cur_image_relevance - cur_image_relevance.min()) / (cur_image_relevance.max() - cur_image_relevance.min())
        image = preprocess_image[0].permute(1, 2, 0).data.cpu().numpy()
        image = (image - image.min()) / (image.max() - image.min())
        vis = show_cam_on_image(image, cur_image_relevance)
        vis = np.uint8(255 * vis)
        vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
        axs[xi][yi].imshow(vis)
        axs[xi][yi].axis('off')
    
    save_image_name = save_path  + os.path.basename(image_path).split('.')[0] + f'_{rtype}' + '.png'
    fig.savefig(save_image_name, dpi=30)
    print("save at: ", save_image_name)
    return


if __name__ == '__main__':
    # llava_model_path = '/home/yaolinli/code/LLaVA/official_ckpts/llava-v1.5-7b-lora'
    # llava_model_path = '/home/yaolinli/code/LLaVA/checkpoints/llava-v1.5-7b-avgpool2_64-lora'
    # mm_projector_type = 'avgpool2_64'
    llava_model_path = '/home/yaolinli/code/LLaVA/official_ckpts/llava-v1.5-7b'
    mm_projector_type = 'mlp2x_gelu'
    BASELINE = None  # None or 'gradcam' or 'rawattn'
    model_name = llava_model_path.split('/')[-1]
    
    use_qformer = True
    if 'mlp' in mm_projector_type:
        use_qformer = False
    # constant settings
    vicuna_model_path = '/home/shared/LargeLanguageModels/vicuna/vicuna-7b-v1.5'
    mm_vision_tower_path = '/home/yaolinli/code/official_ckpts/clip-vit-large-patch14-336'
    pretrain_mm_qformer = '/home/yaolinli/code/official_ckpts/instructblip-vicuna-7b'
    # vit settings from llava: related to attention layers
    mm_vision_select_layer = -2
    select_vit_feature = 'patch'
    version = 'v1'  # v0 means the training stage 1(image caption generation); v1 means the training stage 2 (instruction tuning)
    
    
    '''
    load llava model
    '''
    # lora
    if 'lora' in llava_model_path.lower():
        model = ModelUsage(llava_model_path, vicuna_model_path, mm_vision_tower_path, mm_projector_type, pretrain_mm_qformer, conv_mode='vicuna_v1', version=version, device='cuda')
    else:
    # full mdoel
        model = ModelUsage(llava_model_path, None, mm_vision_tower_path, mm_projector_type, None, conv_mode='vicuna_v1', version=version, device='cuda')
        
    '''
    Input examples
    '''
    # image_file = './llava/explainability/examples/COCO_train2014_000000433723/COCO_train2014_000000433723.jpg'
    image_file = '/home/yaolinli/dataset/coco_caption/COCO_train2014_000000433723.jpg'
    answer = 'A giraffe is standing in the forest.'
    question = "please describe the image briefly."
    target_word = 'giraffe'
    # question = "where is the giraffe standing?"
    # outputs, sent_input_ids = model(image_file, question, answer)
    # outputs['logits'] = torch.Size([1, 589, 32000])
    # outputs['attentions'][0] = torch.Size([1, 32, 589, 589])
    #   — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length)
    
    '''
    Calculate GAE relevance map
    '''
    
    
    '''
    examples target_word = 'giraffe':
    'giraffe' -> 'g ira ffe'
    sent_input_ids: [1, -200, 319, 330, 3055, 17615, 338, 380, 9382, 297, 278, 13569, 29889, 29871]
    word_input_ids: [330, 3055, 17615]
    start_idx = 3
    '''
    sent_input_ids = model.get_input_ids(question, answer)
    # torch.where(sent_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
    image_token_start_index = torch.where(sent_input_ids == IMAGE_TOKEN_INDEX)[1].tolist()[0]
    sent_input_ids = sent_input_ids[0].cpu().numpy().tolist()   # [1, 64]
    answer_input_ids = sent_input_ids[image_token_start_index:]
    word_input_ids = model.tokenizer.encode(target_word)[1:]    # [330, 3055, 17615]
    start_idx = find_indices(sent_input_ids, word_input_ids)[0] - 1  # left shift one position to get the start index of the target word
    
    gen = Generator(model, select_vit_feature, mm_vision_select_layer)
    
    
    if BASELINE is not None:
        rel_map = get_baseline(gen, start_idx, word_input_ids, image_token_start_index,  image_file, question, answer, use_qformer, baseline=BASELINE, last_word=True)
    else:
        # get rel_map for R-GAE and other baselines e.g. gradcam and rawattn
        rel_map = get_rgae(gen, start_idx, word_input_ids, image_token_start_index, image_file, question, answer, use_qformer, last_word=True)
            
    # visualize the relevance map
    save_path = f"/home/yaolinli/code/llava_explain/llava/explainability/examples/rebuttals/S-{BASELINE}_{model_name}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    preprocess_img = model.preprocess_image(image_file)
    show_image_relevance(image_file, rel_map['t2i'], preprocess_img, rtype='t2i', save_path=save_path, target_cap=target_word, sname=BASELINE)
    

        

    if use_qformer:
        R_q_i_aggr = torch.matmul(rel_map['q2i'], rel_map['i2i'])
        show_query_heads(image_file, R_q_i_aggr, preprocess_img, rtype=f'{mm_projector_type}-{target_word}-q2i', save_path=save_path)
        # for qi in range(R_q_i_aggr.shape[0]):
        #     show_image_relevance(image_file, R_q_i_aggr[qi], preprocess_img, rtype='q2i', query_idx=qi)
        
        #     show_image_relevance(image_file, gmap['q2i'][qi], preprocess_img, rtype=f'{BASELINE}-q2i', query_idx=qi)