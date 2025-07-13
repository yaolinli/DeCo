import numpy as np
import torch
from torch.nn.functional import softmax
import pdb
import argparse
import os
import json
from tqdm import tqdm
import shortuuid
import torch.nn.functional as F


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


def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration
    num_tokens = all_layer_matrices[0].shape[1]
    eye = torch.eye(num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    all_layer_matrices = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
                          for i in range(len(all_layer_matrices))]
    matrices_aug = all_layer_matrices
    joint_attention = matrices_aug[start_layer]
    for i in range(start_layer+1, len(matrices_aug)):
        joint_attention = matrices_aug[i].matmul(joint_attention)
    return joint_attention

# rule 5 from paper
def avg_heads(cam, grad, casual=False):
    cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    cam = grad * cam  # [16, 577, 577]
    cam = cam.clamp(min=0).mean(dim=0)  # [577, 577]
    return cam

# rules 6 + 7 from paper
def apply_self_attention_rules(R_ss, R_sq, cam_ss):
    R_sq_addition = torch.matmul(cam_ss, R_sq)
    R_ss_addition = torch.matmul(cam_ss, R_ss)
    return R_ss_addition, R_sq_addition

# rule 10 from paper
def apply_mm_attention_rules(R_ss, R_qq, cam_sq, apply_normalization=True, apply_self_in_rule_10=True):
    R_ss_normalized = R_ss
    R_qq_normalized = R_qq
    if apply_normalization:
        R_ss_normalized = handle_residual(R_ss)
        R_qq_normalized = handle_residual(R_qq)
    R_sq_addition = torch.matmul(R_ss_normalized.t(), torch.matmul(cam_sq, R_qq_normalized))
    if not apply_self_in_rule_10:
        R_sq_addition = cam_sq
    R_sq_addition[torch.isnan(R_sq_addition)] = 0
    return R_sq_addition

# normalization- eq. 8+9
def handle_residual(orig_self_attention):
    self_attention = orig_self_attention.clone()
    diag_idx = range(self_attention.shape[-1])
    self_attention -= torch.eye(self_attention.shape[-1]).to(self_attention.device)
    assert self_attention[diag_idx, diag_idx].min() >= 0
    self_attention = self_attention / self_attention.sum(dim=-1, keepdim=True)
    self_attention += torch.eye(self_attention.shape[-1]).to(self_attention.device)
    return self_attention


# get the mapping matrix from Q*Q -> V*V for the AdaptiveAvgPool2d
def get_mapping_matrix(in_dim, out_dim):
    stride = in_dim // out_dim
    kernel_size = in_dim - (out_dim - 1) * stride

    mapping_matrix = torch.zeros((out_dim * out_dim, in_dim * in_dim), dtype=torch.int)
    
    for i in range(out_dim):
        for j in range(out_dim):
            h_start = i * stride
            w_start = j * stride
            h_end = h_start + kernel_size
            w_end = w_start + kernel_size
            
            # Correctly compute indices for flattened input matrix
            for h in range(h_start, h_end):
                input_start_index = h * in_dim + w_start
                input_end_index = h * in_dim + w_end
                row_index = i * out_dim + j
                mapping_matrix[row_index, input_start_index:input_end_index] = 1    
    scale = stride ** 2
    scale_mapping_matrix = mapping_matrix / scale
    return scale_mapping_matrix


class Generator:
    def __init__(self, model, select_vit_feature='patch', mm_vision_select_layer=-2):
        self.model_usage = model
        self.use_qformer = True
        self.mm_projector_type = self.model_usage.mm_projector_type
        self.select_vit_feature = select_vit_feature
        self.mm_vision_select_layer = mm_vision_select_layer
        self.inter_layers = False
        if 'mlp' in self.model_usage.mm_projector_type:
            self.use_qformer = False
        

    def forward(self, input_ids, labels, images, output_attentions=True):
        outputs = self.model_usage(
            input_ids=input_ids,
            images=images,
            labels=labels,
            attention_mask=input_ids.ne(self.model_usage.tokenizer.pad_token_id),
            output_attentions=output_attentions,
            )
        return outputs

    def generate_transformer_att(self, img, target_index, index=None):
        outputs = self.model_usage(img)
        kwargs = {"alpha": 1,
                  "target_index": target_index}

        if index == None:
            index = outputs['pred_logits'][0, target_index, :-1].max(1)[1]

        kwargs["target_class"] = index

        one_hot = torch.zeros_like(outputs['pred_logits']).to(outputs['pred_logits'].device)
        one_hot[0, target_index, index] = 1
        one_hot_vector = one_hot.clone().detach()
        one_hot.requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * outputs['pred_logits'])

        self.model_usage.zero_grad()
        one_hot.backward(retain_graph=True)

        # self.model_usage.relprop(one_hot_vector, **kwargs)

        decoder_blocks = self.model_usage.transformer.decoder.layers
        encoder_blocks = self.model_usage.transformer.encoder.layers

        # initialize relevancy matrices
        image_bboxes = encoder_blocks[0].self_attn.get_attn().shape[-1]
        query_num = decoder_blocks[0].self_attn.get_attn().shape[-1]

        # image self attention matrix
        self.R_i_i = torch.eye(image_bboxes, image_bboxes).to(encoder_blocks[0].self_attn.get_attn().device)
        # queries self attention matrix
        self.R_q_q = torch.eye(query_num, query_num).to(encoder_blocks[0].self_attn.get_attn().device)
        # impact of image boxes on queries
        self.R_q_i = torch.zeros(query_num, image_bboxes).to(encoder_blocks[0].self_attn.get_attn().device)

        # R_q_i generated from last layer
        decoder_last = decoder_blocks[-1]
        cam_q_i = decoder_last.multihead_attn.get_attn_cam().detach()
        grad_q_i = decoder_last.multihead_attn.get_attn_gradients().detach()
        cam_q_i = avg_heads(cam_q_i, grad_q_i)
        self.R_q_i = cam_q_i
        aggregated = self.R_q_i.unsqueeze_(0)

        aggregated = aggregated[:, target_index, :].unsqueeze_(0)
        return aggregated

    def handle_self_attention_image(self, blocks):
        for i, blk in enumerate(blocks[:self.mm_vision_select_layer+1]):
            grad = blk.self_attn.get_attn_gradients().detach()
                
            if self.use_lrp:
                cam = blk.self_attn.get_attn_cam().detach()
            else:
                cam = blk.self_attn.get_attn().detach()  # [16, 577, 577]
            cam = avg_heads(cam, grad)  # [577, 577]
            if self.select_vit_feature == 'patch':
                # delete cls token 
                cam = cam[1:, 1:]  # [576, 576]
            cam = cam.float()
            self.R_i_i += torch.matmul(cam, self.R_i_i)
            if self.inter_layers:
                self.R_i_i_layers.append((i, self.R_i_i.clone()))

    def handle_self_attention_llm(self, blocks):
        for i, blk in enumerate(blocks):
            grad = blk.self_attn.get_attn_gradients().detach()  # [1, 32, 639, 639]
            if self.use_lrp:
                cam = blk.self_attn.get_attn_cam().detach()
            else:
                cam = blk.self_attn.get_attn().detach()
            # print("llm blk: {};\t cam[99:,:].sum: {};\t grad[99:,:].sum: {}".format(i, cam[0,:,99:,:].sum(), grad[0,:,99:,:].sum()))
            cam = avg_heads(cam, grad)  # [1, 32, 639, 639]
            cam = cam.float()
            self.R_qt_qt += torch.matmul(cam, self.R_qt_qt)
            if self.inter_layers:
                self.R_qt_qt_layers.append((i, self.R_qt_qt.clone()))
            
    def handle_co_attn_self_query(self, block):
        grad = block.attention.attention.get_attn_gradients().detach()  # [1, 32, 64, 64]
        if self.use_lrp:
            cam = block.attention.attention.get_attn_cam().detach()
        else:
            cam = block.attention.attention.get_attn().detach()  # [1, 32, 64, 64]
        cam = avg_heads(cam, grad)  # [64, 64]
        cam = cam.float()
        R_q_q_add, R_q_i_add = apply_self_attention_rules(self.R_q_q, self.R_q_i, cam)
        self.R_q_q += R_q_q_add
        self.R_q_i += R_q_i_add

    def handle_co_attn_query(self, block):
        if self.use_lrp:
            cam_q_i = block.crossattention.attention.get_attn_cam().detach()
        else:
            cam_q_i = block.crossattention.attention.get_attn().detach()  # [1, 32, 64, 576]
        grad_q_i = block.crossattention.attention.get_attn_gradients().detach()  # [1, 32, 64, 576]
        cam_q_i = avg_heads(cam_q_i, grad_q_i)  # [64, 576]
        cam_q_i = cam_q_i.float()
        self.R_q_i += apply_mm_attention_rules(self.R_q_q, self.R_i_i, cam_q_i, apply_normalization=self.normalize_self_attention,  apply_self_in_rule_10=self.apply_self_in_rule_10)

    def register_vision_hooks(self):
        self.vision_attentions = []
        self.vision_gradients = []

        def forward_hook(module, input, output):
            self.vision_attentions.append(output[1])

        def backward_hook(module, grad_in, grad_out):
            self.vision_gradients.append(grad_out[0])

        for layer in self.clip_encoder.layers:
            layer.self_attn.register_forward_hook(forward_hook)
            layer.self_attn.register_backward_hook(backward_hook)
        return self.vision_attentions, self.vision_gradients
    
    def register_llm_hooks(self):
        self.llm_attentions = []
        self.llm_gradients = []

        def forward_hook(module, input, output):
            self.llm_attentions.append(output[1])

        def backward_hook(module, grad_in, grad_out):
            self.llm_gradients.append(grad_out)

        for layer in self.llama_decoder.layers:
            layer.self_attn.register_forward_hook(forward_hook)
            layer.self_attn.register_full_backward_hook(backward_hook)
        return self.llm_attentions, self.llm_gradients
    
    def register_align_hooks(self):
        self.align_self_attentions = []
        self.align_cross_attentions = []
        self.align_self_gradients = []
        self.align_cross_gradients = []

        def self_attention_forward_hook(module, input, output):
            self.align_self_attentions.append(output)

        def cross_attention_forward_hook(module, input, output):
            self.align_cross_attentions.append(output)

        def self_attention_backward_hook(module, grad_in, grad_out):
            self.align_self_gradients.append(grad_out)

        def cross_attention_backward_hook(module, grad_in, grad_out):
            self.align_cross_gradients.append(grad_out)

        for layer in self.mm_projector.layer:
            layer.attention.attention.register_forward_hook(self_attention_forward_hook)
            layer.attention.attention.register_full_backward_hook(self_attention_backward_hook)
            if hasattr(layer, 'crossattention'):
                layer.crossattention.attention.register_forward_hook(cross_attention_forward_hook)
                layer.crossattention.attention.register_full_backward_hook(cross_attention_backward_hook)

        return self.align_self_attentions, self.align_cross_attentions, self.align_self_gradients, self.align_cross_gradients


    def generate_ours(self, img, question, answer, target_index=None, index=None, normalize_self_attention=True, apply_self_in_rule_10=True, inter_layers=False):
        self.inter_layers = inter_layers
        # record R_i_i, R_q_i, R_qt_qt across intermedia layers
        if self.inter_layers:
            self.R_i_i_layers = []
            # self.R_q_i_layers = []
            self.R_qt_qt_layers = []
        
        self.normalize_self_attention = normalize_self_attention    # True
        self.apply_self_in_rule_10 = apply_self_in_rule_10  # True, cross-attention formula（10）
        
        self.clip_encoder = self.model_usage.model.model.vision_tower.vision_tower.vision_model.encoder
        if self.use_qformer and 'qformer' in self.mm_projector_type:
            self.mm_projector = self.model_usage.model.model.mm_projector.qformer.encoder
        self.llama_decoder = self.model_usage.model.model
        
        
        for para in self.model_usage.model.parameters():
            para.requires_grad_(True)
        # for para in self.clip_encoder.parameters():
        #     para.requires_grad_(True)
        # if self.use_qformer:
        #     for para in self.mm_projector.parameters():
        #         para.requires_grad_(True)
            
        # torch.autograd.set_detect_anomaly(True)
        outputs, input_ids = self.model_usage(img, question, answer)
        # self.mm_projector.layer[1].attention.attention.get_attn()
        # self.mm_projector.layer[1].crossattention.attention.get_attn()
        # self.clip_encoder.layers[0].self_attn.get_attn()
        
        output_attentions = outputs['attentions']  # Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).
        output_logits = outputs['logits']  # (torch.FloatTensor of shape (batch_size, sequence_length, config.vocab_size)) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        self.output_logits = output_logits  # torch.Size([1, 589, 32000])
        self.output_attentions = output_attentions
        self.use_lrp = False
        
        
        if index is None:  # get the predict word index of max logit 
            max_value, max_index = torch.max(output_logits[0, index, :], 0)
            index = max_index.item()

        device = output_logits.device
        one_hot = torch.zeros_like(output_logits).to(device)  # (batch_size, sequence_length, config.vocab_size)
        image_bboxes = self.model_usage.image_boxes_len   # input image features
        query_num = image_bboxes
        if self.use_qformer:
            try:
                query_num = self.mm_projector.layer[0].crossattention.attention.get_attn().detach().size(2)
            except:
                query_num = int(self.mm_projector_type.split('_')[-1])
        print("query num: ", query_num)

        target_index = target_index + query_num - 1
        one_hot[0, target_index, index] = 1  # [1, 125, 32000]
        one_hot_vector = one_hot.clone().detach()
        one_hot.requires_grad_(True)
        one_hot = torch.sum(one_hot * output_logits)
        
        # the `backward()` method is called with the argument `retain_graph=True`. This argument is used to indicate that the computational graph should be retained after the backward pass. Retaining the graph can be useful when multiple backward passes are required, such as in cases where the gradients need to be accumulated over multiple iterations.      
        # self.model_usage.model = LlavaLlamaForCausalLM()
        # self.model_usage.model.model = LlavaLlamaModel()
        self.model_usage.model.zero_grad()
        
        # one_hot.backward(retain_graph=True)
        one_hot.backward()
        
        # initialize relevancy matrices
        text_tokens = input_ids.size(1)  # input image/query + text tokens num
        llm_tokens = self.output_attentions[0].size(-1)  # 77 = 64 + 13
        # the input_embeds to llama_decoder is replacing DEFAULT_IMG_TOKEN (num=1) to the image features, therefore the llm_tokens is (text_token_num + img_token_num - 1)
        
        # input text tokens len: 14 image patches num 576 query num: 64 input llm(image+text) tokens len: 77
        # print("\ninput text tokens len:", text_tokens, "\nimage patches num", image_bboxes, "\nquery num:", query_num, "\ninput llm(image+text) tokens len:", llm_tokens)
        
        encoder_blocks = self.clip_encoder.layers
        decoder_blocks = self.llama_decoder.layers
        # get image self attention matrix in ViT
        self.R_i_i = torch.eye(image_bboxes, image_bboxes).to(device)  # [576, 576]
        self.handle_self_attention_image(encoder_blocks)
            
        # if have qformer, calculate the attention matrix   Q * V
        if self.use_qformer and 'qformer' in self.mm_projector_type:
            self.R_q_q = torch.eye(query_num, query_num).to(device)  # torch.Size([64, 64])
            # impact of image boxes on queries
            self.R_q_i = torch.zeros(query_num, image_bboxes).to(device)  # torch.Size([64, 576])
            
            # decoder self attention of queries followd by multi-modal attention
            proj_blocks = self.mm_projector.layer
            for i, blk in enumerate(proj_blocks):
                if i == 0:
                    # qformer self attention
                    self.handle_co_attn_self_query(blk)
                    # qformer cross attention
                    self.handle_co_attn_query(blk)
                elif i == 1:
                    # qformer self attention
                    self.handle_co_attn_self_query(blk)
                else:   # TODO: generalize to more layers
                    # need to process the interleaved self-attention and cross-attention
                    print("not support qformer layer number > 2")
        elif self.use_qformer and 'avgpool' in self.mm_projector_type:
            q_dim = int(query_num ** 0.5)  # 12
            i_dim = int(image_bboxes ** 0.5)  # 24
            assert i_dim % q_dim == 0, "Avgpool Projector: compress rate should be integer"
            cam_q_i = get_mapping_matrix(i_dim, q_dim).to(device).to(self.R_i_i.dtype)
            self.R_q_i = cam_q_i   # [64, 576]
        elif self.use_qformer and 'dabstractor' in self.mm_projector_type:
            print("not support dabstractor yet")  # TODO: support DAbstractor
            exit()
        else:
            self.R_q_i = None
            
        # calculate the attention matrix  (Q + T) * (Q + T)
        self.R_qt_qt = torch.eye(llm_tokens, llm_tokens).to(device)   # [639, 639]
        
        self.handle_self_attention_llm(decoder_blocks)
        
        # for i, blk in enumerate(decoder_blocks):
        #     grad = torch.autograd.grad(one_hot, [blk.self_attn.attn_weights], retain_graph=True)[0].detach()
        
        
        # self.R_qt_qt.shape: [639, 639]
        # self.R_i_i.shape: [576, 576]
        # self.R_q_i: [64, 576]
        
        if self.inter_layers:
            return self.R_qt_qt, self.R_i_i, self.R_q_i, self.R_qt_qt_layers, self.R_i_i_layers
        
        return self.R_qt_qt, self.R_i_i, self.R_q_i


    def generate_raw_attn(self, img, question, answer, target_index=None, index=None, inter_layers=False):
        self.inter_layers = inter_layers
        # record R_i_i, R_q_i, R_qt_qt across intermedia layers
        if self.inter_layers:
            self.R_i_i_layers = []
            # self.R_q_i_layers = []
            self.R_qt_qt_layers = []
            
        for para in self.model_usage.model.parameters():
            para.requires_grad_(True)
            
        outputs, input_ids = self.model_usage(img, question, answer)
        output_logits = outputs['logits']
        
        if index is None:  # get the predict word index of max logit 
            max_value, max_index = torch.max(output_logits[0, index, :], 0)
            index = max_index.item()

        # get attn cam from last decoder layer
        self.llama_decoder = self.model_usage.model.model
        decoder_blocks = self.llama_decoder.layers
        self.clip_encoder = self.model_usage.model.model.vision_tower.vision_tower.vision_model.encoder
        
        cam_qt_qt = decoder_blocks[-1].self_attn.get_attn().detach()  # [1, 32, 125, 125]

        self.cam_qt_qt = cam_qt_qt.mean(dim=1).squeeze(0)   # [125, 125]
        self.cam_q_i = None
        if self.use_qformer:
            self.mm_projector = self.model_usage.model.model.mm_projector.qformer.encoder
            proj_blocks = self.mm_projector.layer
            cam_q_i = proj_blocks[-2].crossattention.attention.get_attn().detach()  # [1, 32, 576, 576]
            self.cam_q_i = cam_q_i.mean(dim=1).squeeze(0)   # [1, 32, 576, 576]
        
        cam_i_i = self.clip_encoder.layers[self.mm_vision_select_layer].self_attn.get_attn().detach().unsqueeze(0)
        self.G_i_i = cam_i_i.mean(dim=1).squeeze(0)[1:, 1:]   # [576, 576]
        return self.cam_qt_qt, self.cam_q_i, self.G_i_i


    def generate_partial_lrp(self, img, target_index, index=None):
        outputs = self.model_usage(img)
        kwargs = {"alpha": 1,
                  "target_index": target_index}

        if index == None:
            index = outputs['pred_logits'][0, target_index, :-1].max(1)[1]

        kwargs["target_class"] = index

        one_hot = torch.zeros_like(outputs['pred_logits']).to(outputs['pred_logits'].device)
        one_hot[0, target_index, index] = 1
        one_hot_vector = one_hot.clone().detach()

        # self.model_usage.relprop(one_hot_vector, **kwargs)

        # get cross attn cam from last decoder layer
        cam_q_i = self.model_usage.transformer.decoder.layers[-1].multihead_attn.get_attn_cam().detach()
        cam_q_i = cam_q_i.reshape(-1, cam_q_i.shape[-2], cam_q_i.shape[-1])
        cam_q_i = cam_q_i.mean(dim=0)
        self.R_q_i = cam_q_i

        # normalize to get non-negative cams
        self.R_q_i = (self.R_q_i - self.R_q_i.min()) / (self.R_q_i.max() - self.R_q_i.min())
        aggregated = self.R_q_i.unsqueeze_(0)

        aggregated = aggregated[:, target_index, :].unsqueeze_(0)
        return aggregated


    def generate_rollout(self, img, target_index):
        outputs = self.model_usage(img)

        decoder_blocks = self.model_usage.transformer.decoder.layers
        encoder_blocks = self.model_usage.transformer.encoder.layers

        cams_image = []
        cams_queries = []
        # image self attention in the encoder
        for blk in encoder_blocks:
            cam = blk.self_attn.get_attn().detach()
            cam = cam.mean(dim=0)
            cams_image.append(cam)

        # decoder self attention of queries
        for blk in decoder_blocks:
            # decoder self attention
            cam = blk.self_attn.get_attn().detach()
            cam = cam.mean(dim=0)
            cams_queries.append(cam)

        # rollout for self-attention values
        self.R_i_i = compute_rollout_attention(cams_image)
        self.R_q_q = compute_rollout_attention(cams_queries)

        decoder_last = decoder_blocks[-1]
        cam_q_i = decoder_last.multihead_attn.get_attn().detach()
        cam_q_i = cam_q_i.reshape(-1, cam_q_i.shape[-2], cam_q_i.shape[-1])
        cam_q_i = cam_q_i.mean(dim=0)
        self.R_q_i = torch.matmul(self.R_q_q.t(), torch.matmul(cam_q_i, self.R_i_i))
        aggregated = self.R_q_i.unsqueeze_(0)

        aggregated = aggregated[:, target_index, :].unsqueeze_(0)
        return aggregated

    def gradcam(self, cam, grad):
        cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])  # [32, 125, 125]
        grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])  # [32, 125, 125]
        grad = grad.mean(dim=[1, 2], keepdim=True)  # [32, 1, 1]
        cam = (cam * grad).mean(0).clamp(min=0)
        return cam.float()

    def generate_attn_gradcam(self, img, question, answer, target_index=None, index=None, inter_layer=False):
        
        for para in self.model_usage.model.parameters():
            para.requires_grad_(True)
        
        self.mm_projector = self.model_usage.model.model.mm_projector
        if self.use_qformer and 'cabstractor' in self.mm_projector_type:
            activations = {}
            gradients = {}

            def get_activations_hook(layer_name):
                def hook(module, input, output):
                    activations[layer_name] = output.detach()
                return hook

            def get_gradients_hook(layer_name):
                def hook(module, input_grad, output_grad):
                    gradients[layer_name] = output_grad[0].detach()
                return hook

            first_resblk = self.mm_projector.projector.net[0].b3.conv3 
            second_resblk = self.mm_projector.projector.net[2].b3.conv3
            # register hook
            first_resblk.register_forward_hook(get_activations_hook('first_resblk'))
            first_resblk.register_backward_hook(get_gradients_hook('first_resblk'))
            second_resblk.register_forward_hook(get_activations_hook('second_resblk'))
            second_resblk.register_backward_hook(get_gradients_hook('second_resblk'))

        
        # forward pass
        outputs, input_ids = self.model_usage(img, question, answer)
        output_logits = outputs['logits']
        
        if index is None:  # get the predict word index of max logit 
            max_value, max_index = torch.max(output_logits[0, index, :], 0)
            index = max_index.item()

        device = output_logits.device
        one_hot = torch.zeros_like(output_logits).to(device)  # (batch_size, sequence_length, config.vocab_size)
        
        image_bboxes = self.model_usage.image_boxes_len   # input image features
        query_num = image_bboxes
        if self.use_qformer:
            try:
                query_num = self.mm_projector.layer[0].crossattention.attention.get_attn().detach().size(2)
            except:
                query_num = int(self.mm_projector_type.split('_')[-1])
        print("query num: ", query_num)
        
            
        target_index = target_index + query_num - 1
        one_hot[0, target_index, index] = 1  # [1, 125, 32000]
        one_hot_vector = one_hot.clone().detach()
        one_hot.requires_grad_(True)
        one_hot = torch.sum(one_hot * output_logits)

        self.model_usage.model.zero_grad()
        one_hot.backward()


        # get attn cam from last decoder layer
        self.llama_decoder = self.model_usage.model.model
        decoder_blocks = self.llama_decoder.layers
        self.clip_encoder = self.model_usage.model.model.vision_tower.vision_tower.vision_model.encoder
        encoder_blocks = self.clip_encoder.layers
        
        cam_qt_qt = decoder_blocks[-1].self_attn.get_attn().detach()  # [1, 32, 125, 125]
        grad_qt_qt = decoder_blocks[-1].self_attn.get_attn_gradients().detach()
        cam_qt_qt = self.gradcam(cam_qt_qt, grad_qt_qt)   
        self.G_qt_qt = cam_qt_qt
        
        cam_i_i = encoder_blocks[self.mm_vision_select_layer].self_attn.get_attn().detach()
        grad_i_i = encoder_blocks[self.mm_vision_select_layer].self_attn.get_attn_gradients().detach()
        cam_i_i = self.gradcam(cam_i_i, grad_i_i)
        self.G_i_i = cam_i_i[1:, 1:]   # [576, 576]
        
        # print("cam_i_i contains NaN:", torch.isnan(cam_i_i).any().item())
        # print("self.G_i_i contains NaN:", torch.isnan(self.G_i_i).any().item())
        # if torch.isnan(cam_i_i).any().item():
        #     print('cam_i_i has nan')
        # if torch.isnan(self.G_i_i).any().item():
        #     print('G_i_i has nan')
            
        self.G_q_i = None
        if self.use_qformer:
            if 'qformer' in self.mm_projector_type:
                self.mm_projector = self.model_usage.model.model.mm_projector.qformer.encoder
                proj_blocks = self.mm_projector.layer
                cam_q_i = proj_blocks[-2].crossattention.attention.get_attn().detach()  # [1, 32, 576, 576]
                grad_q_i = proj_blocks[-2].crossattention.attention.get_attn_gradients().detach()
                cam_q_i = self.gradcam(cam_q_i, grad_q_i)
                self.G_q_i = cam_q_i
            elif 'avgpool' in self.mm_projector_type:
                q_dim = int(int(self.mm_projector_type.split('_')[-1]) ** 0.5)  # 12
                i_dim = int(image_bboxes ** 0.5)  # 24
                cam_q_i = get_mapping_matrix(i_dim, q_dim).to(self.G_i_i.device).to(self.G_i_i.dtype)
                self.G_q_i = cam_q_i
            # elif 'maxpool' in self.mm_projector_type:
            #     continue
            elif 'cabstractor' in self.mm_projector_type:
                # print(activations)
                # print(gradients)
                # get q_i mapping matrix from adaptive AvgPool
                q_dim = int(int(self.mm_projector_type.split('_')[-1]) ** 0.5)  # 12
                i_dim = int(self.G_i_i.shape[0] ** 0.5)  # 24
                cam_avg_q_i = get_mapping_matrix(i_dim, q_dim).to(self.G_i_i.device).to(self.G_i_i.dtype)  # [64, 576]
                # get resnet block i*i gradcam matrix
                cam_resblk_i_i = None
                cam_r_i_i = activations['first_resblk']  # shape: [1, 1024, 24, 24]
                grad_r_i_i = gradients['first_resblk']  # shape: [1, 1024, 24, 24]
                cam_resblk_i_i = self.gradcam(cam_r_i_i, grad_r_i_i)  # torch.Size([24, 24])
                cam_resblk_i_i = cam_resblk_i_i.reshape(-1).unsqueeze(0)  # torch.Size([1, 576])
                # get resnet block q*q gradcam matrix
                cam_resblk_q_q = None
                cam_r_q_q = activations['second_resblk']  # shape: [1, 1024, 8, 8]
                grad_r_q_q = gradients['second_resblk']  # shape: [1, 1024, 8, 8]
                cam_resblk_q_q = self.gradcam(cam_r_q_q, grad_r_q_q)  # torch.Size([8, 8])
                cam_resblk_q_q = cam_resblk_q_q.reshape(-1).unsqueeze(1)  # torch.Size([64, 1])
                
                # normalize
                norm_cam_resblk_q_q = normalize_rgae(cam_resblk_q_q, 'uniform_0_1')
                norm_cam_resblk_i_i = normalize_rgae(cam_resblk_i_i, 'uniform_0_1')
                norm_cam_resblk_q_q = norm_cam_resblk_q_q / norm_cam_resblk_q_q.sum()
                norm_cam_resblk_i_i = norm_cam_resblk_i_i / norm_cam_resblk_i_i.sum()
                # final cam_q_i
                cam_q_i = norm_cam_resblk_q_q * cam_avg_q_i * norm_cam_resblk_i_i
                self.G_q_i = cam_q_i
                # print(f"cam_resblk_q_q.sum: {norm_cam_resblk_q_q.sum().item()};\t cam_avg_q_i.sum: {cam_avg_q_i.sum().item()};\t cam_resblk_i_i.sum: {norm_cam_resblk_i_i.sum().item()}\t")
                # print("self.G_q_i.sum():", self.G_q_i.sum().item())
                # print("self.G_i_i.sum():", self.G_i_i.sum().item())
            else:
                print("projctor type not supported")

        if torch.isnan(self.G_qt_qt).any().item():
            print("self.G_qt_qt contains NaN in func generate_attn_gradcam")
        if torch.isnan(self.G_q_i).any().item():
            print("self.G_q_i contains NaN in func generate_attn_gradcam")
        if torch.isnan(self.G_i_i).any().item():
            print("self.G_i_i contains NaN in func generate_attn_gradcam")
        return self.G_qt_qt, self.G_q_i, self.G_i_i




