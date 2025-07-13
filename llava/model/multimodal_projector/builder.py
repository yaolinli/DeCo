import torch
import torch.nn as nn
import re
from typing import Optional
import pdb
import math
from transformers import PretrainedConfig, Blip2PreTrainedModel, Blip2Config, Blip2QFormerModel, Blip2QFormerConfig
from einops import rearrange
from timm.models.layers import LayerNorm, LayerNorm2d
from timm.models.regnet import RegStage
from functools import partial
from transformers.configuration_utils import PretrainedConfig
from transformers.models.deformable_detr import DeformableDetrConfig
from transformers.models.deformable_detr.modeling_deformable_detr import (
    DeformableDetrDecoder,
    DeformableDetrDecoderLayer,
    DeformableDetrDecoderOutput,
)

class MaxPoolProjector(nn.Module):
    def __init__(
        self,
        layer_num: int = 2,
        query_num: int = 144,
        mm_hidden_size: int = 1024,
        llm_hidden_size: int = 4096,
    ):
        super().__init__()
        self.layer_num = layer_num
        self.query_num = query_num
        self.mm_hidden_size = mm_hidden_size
        self.llm_hidden_size = llm_hidden_size
        self.build_net()
        
    def build_net(self):
        hw = int(self.query_num ** 0.5)
        sampler = nn.AdaptiveMaxPool2d((hw, hw))
        self.sampler = sampler
        modules = [nn.Linear(self.mm_hidden_size, self.llm_hidden_size)]
        for _ in range(1, self.layer_num):
            modules.append(nn.GELU())
            modules.append(nn.Linear(self.llm_hidden_size, self.llm_hidden_size))
        self.mlp_projector = nn.Sequential(*modules)
        
    def forward(self, visual_feat: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, h_dim = visual_feat.shape  # 576
        hw = int(seq_len ** 0.5)  # 24
        shaped_visual_feat = rearrange(visual_feat, "b (h w) d -> b d h w", h=hw, w=hw)  # torch.Size([64, 1024, 24, 24])
        pooled_visual_feat = self.sampler(shaped_visual_feat)  # torch.Size([64, 1024, 12, 12])
        reshaped_visual_feat = rearrange(pooled_visual_feat, "b d h w -> b (h w) d")  # [64, 144, 1024]
        output_feat = self.mlp_projector(reshaped_visual_feat)  # [64, 144, 4096])
        return output_feat

class HoneybeeVisualProjectorConfig(PretrainedConfig):
    model_type = "mllm_visual_projector"
    def __init__(
        self,
        projector_type: str = "c-abstractor",
        hidden_size: int = 1024,  #
        num_hidden_layers: int = 6,  #
        num_attention_heads: int = 16,  #
        intermediate_size: int = 4096,  #
        attention_probs_dropout_prob: float = 0.1,  #
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-6,  #
        encoder_hidden_size: int = 1024,  # This will be overwritten by vision_model's hidden_size
        depth: int = 3,  # the layer num of ResNet Block
        mlp_depth: int = 2,  # the layer num of the MLP Block that mapping projector dim to llm dim
        num_queries: int = 144,
        pos_emb=True,
        feature_layer_index=-2,  # vision feature layer index; -1: last layer
        num_eos_tokens=0,
        use_cls=True,
        prenorm=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.depth = depth
        self.mlp_depth = mlp_depth
        self.num_queries = num_queries
        self.projector_type = projector_type
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.encoder_hidden_size = encoder_hidden_size

        self.pos_emb = pos_emb
        self.feature_layer_index = feature_layer_index
        self.num_eos_tokens = num_eos_tokens
        self.use_cls = use_cls
        self.prenorm = prenorm
        self.eval = False  # train stage or eval stage
    
    
def build_pos_embeds(
    config: HoneybeeVisualProjectorConfig, num_input_tokens: int, vision_hidden_size: int
):
    # pos emb
    if config.pos_emb:
        pos_emb = torch.zeros(1, num_input_tokens, vision_hidden_size, dtype=torch.float16).to("cuda")
        pos_emb = torch.nn.Parameter(pos_emb, requires_grad=False)
        nn.init.trunc_normal_(pos_emb, mean=0.0, std=0.02)

        # pos_emb = torch.nn.Parameter(torch.zeros(1, num_input_tokens, vision_hidden_size), requires_grad=False)
        # nn.init.trunc_normal_(pos_emb, mean=0.0, std=0.02)
    else:
        pos_emb = None

    return pos_emb


def build_eos_tokens(config: HoneybeeVisualProjectorConfig, output_hidden_size: int):
    # think tokens
    num_eos_tokens = config.num_eos_tokens
    if num_eos_tokens:
        eos_tokens = torch.nn.Parameter(torch.randn(1, num_eos_tokens, output_hidden_size))
        nn.init.trunc_normal_(eos_tokens, mean=0.0, std=config.initializer_range)
    else:
        eos_tokens = None

    return eos_tokens


def build_prenorm(config: HoneybeeVisualProjectorConfig):
    if config.prenorm:
        prenorm = LayerNorm(config.encoder_hidden_size)
    else:
        prenorm = None
    return prenorm


def build_mlp(depth, hidden_size, output_hidden_size):
    layers = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        layers.append(nn.SiLU())
        layers.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*layers)


class DAbstractor(DeformableDetrDecoder):
    # reference: https://github.com/huggingface/transformers/blob/v4.34.1/src/transformers/models/deformable_detr/modeling_deformable_detr.py#1279
    def __init__(
        self, config: DeformableDetrConfig, num_input_tokens: int, output_hidden_size: int, *igargs
    ):
        super().__init__(config)
        self.projector_type = config.projector_type
        self.except_cls = config.except_cls
        self.num_queries = config.num_queries
        self.num_input_tokens = num_input_tokens

        self.num_feature_levels = config.num_feature_levels
        self.isMs = self.num_feature_levels > 1

        self.layers = nn.ModuleList(
            [DeformableDetrDecoderLayer(config) for _ in range(config.decoder_layers)]
        )

        # define input projection layers
        is_dim_missmatch = config.d_model != config.encoder_hidden_size
        input_proj_list = []
        for _ in range(self.num_feature_levels):
            if is_dim_missmatch:
                # All hidden dims for output of each layer are the same in the CLIP vision encoder.
                input_proj_list.append(nn.Linear(config.encoder_hidden_size, config.d_model))
            else:
                input_proj_list.append(nn.Identity())

        self.input_proj = nn.ModuleList(input_proj_list)

        # define level_emb layer
        if self.isMs:  # for multi-scale features
            assert config.num_feature_levels == len(config.feature_layer_index)
            self.level_emb = nn.Parameter(
                torch.Tensor(1, config.num_feature_levels, 1, config.d_model)
            )
            nn.init.normal_(self.level_emb)  # same initialize with the original implementation

        # initialize the query embeddings as pooled visual feature map
        self.pooled_v_target = config.pooled_v_target
        if self.pooled_v_target != "none":
            tgt_hw = int(config.num_queries**0.5)
            self.downsampler = nn.AdaptiveAvgPool2d((tgt_hw, tgt_hw))
            self.query_position_embeddings = nn.Embedding(config.num_queries, config.d_model)
        else:
            self.query_position_embeddings = nn.Embedding(config.num_queries, config.d_model * 2)

        # define reference points
        # manual initialization + make them as learable parameters
        valid_ratios_q, spatial_shapes_q, _ = self._prepare_ddetr_inputs(1, num_input_tokens, 1)
        reference_points = self._get_query_reference_points(spatial_shapes_q, valid_ratios_q)
        self.reference_points = nn.Parameter(reference_points)

        # think tokens
        self.eos_tokens = build_eos_tokens(config, config.d_model)

        # pos emb
        if self.except_cls:
            num_input_tokens -= 1
        self.v_pos_emb = build_pos_embeds(config, num_input_tokens, config.d_model)

        # token projector
        if output_hidden_size != config.d_model:
            self.output_proj = nn.Linear(config.d_model, output_hidden_size)
        else:
            self.output_proj = nn.Identity()

    def _get_query_reference_points(self, spatial_shapes, valid_ratios):
        """
        Get reference points for each feature map. Used in decoder.

        Args:
            spatial_shapes (`torch.LongTensor` of shape `(num_feature_levels, 2)`):
                Spatial shapes of each feature map.
            valid_ratios (`torch.FloatTensor` of shape `(batch_size, num_feature_levels, 2)`):
                Valid ratios of each feature map.
            device (`torch.device`):
                Device on which to create the tensors.
        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_queries, num_feature_levels, 2)`
        """
        reference_points_list = []
        steps = int(self.num_queries**0.5)
        for level, (height, width) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, height - 0.5, steps, dtype=torch.float32),
                torch.linspace(0.5, width - 0.5, steps, dtype=torch.float32),
                indexing="ij",
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, level, 1] * height)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, level, 0] * width)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points.squeeze(2)

    def _forward(
        self,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        position_embeddings=None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        valid_ratios=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`):
                The query embeddings that are passed into the decoder.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing cross-attention on padding pixel_values of the encoder. Mask values selected
                in `[0, 1]`:
                - 1 for pixels that are real (i.e. **not masked**),
                - 0 for pixels that are padding (i.e. **masked**).
            position_embeddings (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`, *optional*):
                Position embeddings that are added to the queries and keys in each self-attention layer.
            reference_points (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)` is `as_two_stage` else `(batch_size, num_queries, 2)` or , *optional*):
                Reference point in range `[0, 1]`, top-left (0,0), bottom-right (1, 1), including padding area.
            spatial_shapes (`torch.FloatTensor` of shape `(num_feature_levels, 2)`):
                Spatial shapes of the feature maps.
            level_start_index (`torch.LongTensor` of shape `(num_feature_levels)`, *optional*):
                Indexes for the start of each feature level. In range `[0, sequence_length]`.
            valid_ratios (`torch.FloatTensor` of shape `(batch_size, num_feature_levels, 2)`, *optional*):
                Ratio of valid area in each feature level.

            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is not None:
            hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        intermediate = ()
        intermediate_reference_points = ()

        for _, decoder_layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = (
                    reference_points[:, :, None]
                    * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
                )
            else:
                if reference_points.shape[-1] != 2:
                    raise ValueError("Reference points' last dimension must be of size 2")
                reference_points_input = reference_points[:, :, None] * valid_ratios[:, None]

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    encoder_hidden_states=encoder_hidden_states,
                    reference_points=reference_points_input,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    encoder_attention_mask=encoder_attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            intermediate += (hidden_states,)
            intermediate_reference_points += (reference_points,)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # Keep batch_size as first dimension
        intermediate = torch.stack(intermediate, dim=1)
        intermediate_reference_points = torch.stack(intermediate_reference_points, dim=1)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    intermediate,
                    intermediate_reference_points,
                    all_hidden_states,
                    all_self_attns,
                ]
                if v is not None
            )
        return DeformableDetrDecoderOutput(
            last_hidden_state=hidden_states,
            intermediate_hidden_states=intermediate,
            intermediate_reference_points=intermediate_reference_points,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _process_v_features(self, visual_feat):
        # visual_feat: [B, len, dim] or [B, lvls, len, dim]
        if self.except_cls:
            visual_feat = visual_feat[:, :, 1:] if self.isMs else visual_feat[:, 1:]

        if self.isMs:
            visual_feats = []
            for level in range(self.num_feature_levels):
                visual_feats.append(self.input_proj[level](visual_feat[:, level]))
            visual_feat = torch.stack(visual_feats, 1)

            # add pos emb [1, len, dim]
            if self.v_pos_emb is not None:
                self.v_pos_emb  = self.v_pos_emb.to(visual_feat.device)
                visual_feat = visual_feat + self.v_pos_emb.unsqueeze(1)

            # add lvl emb [1, lvls, 1, dim]
            visual_feat = visual_feat + self.level_emb
            visual_feat = visual_feat.flatten(1, 2)  # [B, lvls, v_len, dim] -> [B, lvls*v_len, dim]
        else:
            visual_feat = self.input_proj[0](visual_feat)
            if self.v_pos_emb is not None:
                self.v_pos_emb  = self.v_pos_emb.to(visual_feat.device)
                visual_feat = visual_feat + self.v_pos_emb

        return visual_feat

    def _convert_dtype_device(self, tgt_feat, dtype=None, device=None):
        # tgt_feat: target tensor to be converted
        _dtype = tgt_feat.dtype if dtype is None else dtype
        _device = tgt_feat.device if device is None else device

        tgt_feat = tgt_feat.type(_dtype).to(_device)

        return tgt_feat

    def _prepare_ddetr_inputs(self, batch_size, seq_len, lvls, dtype=None, device=None):
        # assume there are no paddings in a feature map
        valid_ratios = torch.ones(batch_size, lvls, 2)

        # assume all feature maps have the same sequence length (i.e., the same shape)
        spatial_shapes = torch.tensor([int(seq_len**0.5), int(seq_len**0.5)]).repeat(lvls, 1)
        level_start_index = torch.arange(0, seq_len * lvls, seq_len)

        if dtype is not None and device is not None:
            valid_ratios = self._convert_dtype_device(valid_ratios, dtype=dtype, device=device)
            spatial_shapes = self._convert_dtype_device(
                spatial_shapes, dtype=torch.long, device=device
            )
            level_start_index = self._convert_dtype_device(
                level_start_index, dtype=torch.long, device=device
            )

        return valid_ratios, spatial_shapes, level_start_index

    def _make_pooled_queries(self, visual_feat):
        assert (
            self.num_feature_levels == 1
        )  # currently do not support multi-scale features for the v-pooled Q

        batch_size, seq_len, h_dim = visual_feat.shape
        query_embeds = self.query_position_embeddings.weight
        if self.pooled_v_target != "none":
            hw_v = int(seq_len**0.5)
            hw_q = int(self.num_queries**0.5)
            visual_feat = rearrange(visual_feat, "b (h w) d -> b d h w", h=hw_v, w=hw_v)
            if self.pooled_v_target == "tgt":
                query_embed = query_embeds.unsqueeze(0).expand(batch_size, -1, -1)
                target = self.downsampler(visual_feat)
                target = rearrange(target, "b d h w -> b (h w) d", h=hw_q, w=hw_q)
            else:
                target = query_embeds.unsqueeze(0).expand(batch_size, -1, -1)
                query_embed = self.downsampler(visual_feat)
                query_embed = rearrange(query_embed, "b d h w -> b (h w) d", h=hw_q, w=hw_q)
        else:
            query_embed, target = torch.split(query_embeds, h_dim, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(batch_size, -1, -1)
            target = target.unsqueeze(0).expand(batch_size, -1, -1)

        return query_embed, target

    def forward(self, visual_feat):
        """
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`):
                The query embeddings that are passed into the decoder.
        """
        # deformable attention only supports fp32
        original_dtype = visual_feat.type()
        visual_feat = visual_feat.type(torch.cuda.FloatTensor)
        visual_feat = self._process_v_features(visual_feat)

        batch_size, seq_len, h_dim = visual_feat.shape
        seq_len /= self.num_feature_levels

        query_embed, target = self._make_pooled_queries(visual_feat)
        reference_points = self.reference_points.expand(batch_size, -1, -1)

        valid_ratios, spatial_shapes, level_start_index = self._prepare_ddetr_inputs(
            batch_size, seq_len, self.num_feature_levels, visual_feat.dtype, visual_feat.device
        )

        if "d-abstractor" in self.projector_type:
            self.layers.to(torch.float32)
        target = target.type(torch.cuda.FloatTensor)
        reference_points = reference_points.type(torch.cuda.FloatTensor)
        spatial_shapes = spatial_shapes.type(torch.cuda.LongTensor)
        level_start_index = level_start_index.type(torch.cuda.LongTensor)
        decoder_outputs_dict = self._forward(
            inputs_embeds=target,
            position_embeddings=query_embed,
            encoder_hidden_states=visual_feat,
            valid_ratios=valid_ratios,
            reference_points=reference_points,
            return_dict=True,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )

        decoder_outputs = decoder_outputs_dict.last_hidden_state

        if self.eos_tokens is not None:
            decoder_outputs = torch.cat(
                [decoder_outputs, self.eos_tokens.expand(batch_size, -1, -1)], dim=1
            )

        decoder_outputs = decoder_outputs.type(original_dtype)
        decoder_outputs = self.output_proj(decoder_outputs)
        

        return DeformableDetrDecoderOutput(
            last_hidden_state=decoder_outputs,
        )[0]




class Projector(nn.Module):
    """Base projector class"""
    def __init__(
        self,
        config: HoneybeeVisualProjectorConfig,
        num_input_tokens: int = 576,
        output_hidden_size: int = 4096,
    ):
        super().__init__()
        self.config = config
        self.num_input_tokens = num_input_tokens
        self.output_hidden_size = output_hidden_size

        # think tokens
        self.eos_tokens = build_eos_tokens(config, output_hidden_size)

        # pos emb
        self.pos_emb = build_pos_embeds(config, num_input_tokens, config.encoder_hidden_size)

        self.prenorm = build_prenorm(config)

        self.build_net()

    def build_net(self):
        raise NotImplementedError()

    def _forward(self, x):
        raise NotImplementedError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, encoder_hidden_size) tensor from the visual backbone (CLIP visual encoder), including cls token.
        """
        if self.prenorm is not None:
            x = self.prenorm(x)

        if self.pos_emb is not None:
            self.pos_emb = self.pos_emb.to(x.device)
            self.pos_emb.requires_grad_(x.requires_grad)
            x += self.pos_emb

        x = self._forward(x)  # (B, L, output_hidden_size)

        B = x.size(0)
        if self.eos_tokens is not None:
            x = torch.cat([x, self.eos_tokens.expand(B, -1, -1)], dim=1)
        return x
    

class ConvProjector(Projector):
    def _forward(self, x):
        # x: [B, L, dim]
        # x = x[:, 1:]  # drop cls token and 2d forward
        hw = int(x.size(1) ** 0.5)
        x = rearrange(x, "b (h w) d -> b d h w", h=hw, w=hw)
        x = self.net(x)
        x = rearrange(x, "b d h w -> b (h w) d")
        x = self.readout(x)
        return x


class AvgPoolProjector(nn.Module):
    def __init__(
        self,
        layer_num: int = 2,
        query_num: int = 144,
        mm_hidden_size: int = 1024,
        llm_hidden_size: int = 4096,
    ):
        super().__init__()
        self.layer_num = layer_num
        self.query_num = query_num
        self.mm_hidden_size = mm_hidden_size
        self.llm_hidden_size = llm_hidden_size
        self.build_net()
        
    def build_net(self):
        hw = int(self.query_num ** 0.5)
        sampler = nn.AdaptiveAvgPool2d((hw, hw))
        self.sampler = sampler
        modules = [nn.Linear(self.mm_hidden_size, self.llm_hidden_size)]
        for _ in range(1, self.layer_num):
            modules.append(nn.GELU())
            modules.append(nn.Linear(self.llm_hidden_size, self.llm_hidden_size))
        self.mlp_projector = nn.Sequential(*modules)
        
    def forward(self, visual_feat: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, h_dim = visual_feat.shape  # 576
        hw = int(seq_len ** 0.5)  # 24
        shaped_visual_feat = rearrange(visual_feat, "b (h w) d -> b d h w", h=hw, w=hw)  # torch.Size([64, 1024, 24, 24])
        pooled_visual_feat = self.sampler(shaped_visual_feat)  # torch.Size([64, 1024, 12, 12])
        reshaped_visual_feat = rearrange(pooled_visual_feat, "b d h w -> b (h w) d")  # [64, 144, 1024]
        output_feat = self.mlp_projector(reshaped_visual_feat)  # [64, 144, 4096])
        return output_feat
        

class CAbstractor(ConvProjector):
    """C-Abstractor"""
    def build_net(self):
        encoder_hidden_size = self.config.encoder_hidden_size
        hidden_size = self.config.hidden_size
        output_hidden_size = self.output_hidden_size
        depth = self.config.depth
        mlp_depth = self.config.mlp_depth

        n_queries = self.config.num_queries
        assert (n_queries ** 0.5).is_integer(), "n_queries must be square number"
        hw = int(n_queries ** 0.5)

        # RegBlock = ResBlock + SE
        RegBlock = partial(
            RegStage,
            stride=1,
            dilation=1,
            act_layer=nn.SiLU,
            norm_layer=LayerNorm2d,
        )

        s1 = RegBlock(
            depth,
            encoder_hidden_size,
            hidden_size,
        )
        sampler = nn.AdaptiveAvgPool2d((hw, hw))
        s2 = RegBlock(
            depth,
            hidden_size,
            hidden_size,
        )

        self.net = nn.Sequential(s1, sampler, s2)
        self.readout = build_mlp(mlp_depth, hidden_size, output_hidden_size)


class CAbstractorProjector(nn.Module):
    def __init__(
        self,
        vision_dim: int,
        llm_dim: int,
        num_query_tokens: int = 64,
        depth: int = 3,  # the layer num of ResNet Block
        mlp_depth: int = 2,  # the layer num of the MLP Block that mapping projector dim to llm dim
    ) -> None:
        super().__init__()
        # encoder_hidden_size: int = 1024,  # This will be overwritten by vision_model's hidden_size
        # depth: int = 1,
        # mlp_depth: int = 2,
        # num_queries: int = 144,
        CAbstractor_config = HoneybeeVisualProjectorConfig(projector_type='c-abstractor', encoder_hidden_size=vision_dim, depth=depth, mlp_depth=mlp_depth, num_queries=num_query_tokens)
        self.projector = CAbstractor(CAbstractor_config, output_hidden_size=llm_dim)

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        # img_patches: [16, 576, 1024]
        return self.projector(img_patches)


def DAbstractor_config_template(vision_dim, layer_num, num_query_tokens, projector_type='d-abstractor', except_cls=False):
    DAbstractor_config = type('DeformableDetrConfig', (PretrainedConfig,), {
        "_name_or_path": "",
        "activation_dropout": 0.0,
        "activation_function": "relu",
        "add_cross_attention": False,
        "architectures": None,
        "attention_dropout": 0.0,
        "auxiliary_loss": False,
        "backbone": "resnet50",
        "backbone_config": None,
        "bad_words_ids": None,
        "bbox_cost": 5,
        "bbox_loss_coefficient": 5,
        "begin_suppress_tokens": None,
        "bos_token_id": None,
        "chunk_size_feed_forward": 0,
        "class_cost": 1,
        "cross_attention_hidden_size": None,
        "d_model": 1024,
        "decoder_attention_heads": 8,
        "decoder_ffn_dim": 1024,
        "decoder_layers": layer_num,
        "decoder_n_points": 4,
        "decoder_start_token_id": None,
        "dice_loss_coefficient": 1,
        "dilation": False,
        "disable_custom_kernels": False,
        "diversity_penalty": 0.0,
        "do_sample": False,
        "dropout": 0.1,
        "early_stopping": False,
        "encoder_attention_heads": 8,
        "encoder_ffn_dim": 1024,
        "encoder_hidden_size": vision_dim,
        "encoder_layerdrop": 0.0,
        "encoder_layers": 6,
        "encoder_n_points": 4,
        "encoder_no_repeat_ngram_size": 0,
        "eos_coefficient": 0.1,
        "eos_token_id": None,
        "except_cls": except_cls,
        "exponential_decay_length_penalty": None,
        "feature_layer_index": -2,
        "finetuning_task": None,
        "focal_alpha": 0.25,
        "forced_bos_token_id": None,
        "forced_eos_token_id": None,
        "giou_cost": 2,
        "giou_loss_coefficient": 2,
        "id2label": {
            "0": "LABEL_0",
            "1": "LABEL_1"
        },
        "init_std": 0.02,
        "init_xavier_std": 1.0,
        "initializer_range": 0.02,
        "is_decoder": False,
        "is_encoder_decoder": True,
        "label2id": {
            "LABEL_0": 0,
            "LABEL_1": 1
        },
        "length_penalty": 1.0,
        "mask_loss_coefficient": 1,
        "max_length": 20,
        "max_position_embeddings": 1024,
        "min_length": 0,
        "model_type": "deformable_detr",
        "no_repeat_ngram_size": 0,
        "num_beam_groups": 1,
        "num_beams": 1,
        "num_channels": 3,
        "num_eos_tokens": 0,
        "num_feature_levels": 1,
        "num_queries": num_query_tokens,
        "num_return_sequences": 1,
        "output_attentions": False,
        "output_hidden_states": False,
        "output_scores": False,
        "pad_token_id": None,
        "pooled_v_target": "query",
        "pos_emb": 1,
        "position_embedding_type": "sine",
        "prefix": None,
        "problem_type": None,
        "projector_type": projector_type,
        "pruned_heads": {},
        "remove_invalid_values": False,
        "repetition_penalty": 1.0,
        "return_dict": True,
        "return_dict_in_generate": False,
        "sep_token_id": None,
        "suppress_tokens": None,
        "task_specific_params": None,
        "temperature": 1.0,
        "tf_legacy_loss": False,
        "tie_encoder_decoder": False,
        "tie_word_embeddings": True,
        "tokenizer_class": None,
        "top_k": 50,
        "top_p": 1.0,
        "torch_dtype": None,
        "torchscript": False,
        "transformers_version": "4.30.2",
        "two_stage": False,
        "two_stage_num_proposals": 300,
        "typical_p": 1.0,
        "use_bfloat16": False,
        "use_pretrained_backbone": False,
        "use_timm_backbone": True,
        "with_box_refine": False
    })()
    return DAbstractor_config


class DAbstractorProjector(nn.Module):
    def __init__(
        self,
        vision_dim: int,
        llm_dim: int,
        num_query_tokens: int = 144,
        layer_num: int = 2,  # the layer num of Deformable Detr Decoder
    ) -> None:
        super().__init__()
        DAbstractor_config = DAbstractor_config_template(projector_type='d-abstractor', vision_dim=vision_dim, layer_num=layer_num, num_query_tokens=num_query_tokens)
        self.projector = DAbstractor(DAbstractor_config, num_input_tokens=576, output_hidden_size=llm_dim)

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        # img_patches: [16, 576, 1024]
        return self.projector(img_patches)
    
    
    
class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class CustomBlip2QFormerMultiHeadAttention(nn.Module):
    def __init__(self, config, is_cross_attention=False, save_attention=False):
        super().__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention heads (%d)"
                % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        if is_cross_attention:
            self.key = nn.Linear(config.encoder_hidden_size, self.all_head_size)
            self.value = nn.Linear(config.encoder_hidden_size, self.all_head_size)
        else:
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)
        self.save_attention = True
        self.save_gradient = True
        self.attn = None
        self.attn_gradients = None

    # def save_attn_gradients(self, attn_gradients):
    #     if self.attn_gradients is not None:
    #         self.attn_gradients = [self.attn_gradients, attn_gradients]
    #     else:
    #         self.attn_gradients = attn_gradients

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients
            
    def get_attn_gradients(self):
        return self.attn_gradients

    # def save_attn(self, attn):
    #     if self.attn is not None:
    #         self.attn = [self.attn, attn]
    #     else:
    #         self.attn = attn
    
    def save_attn(self, attn):
        self.attn = attn
            
    def get_attn(self):
        return self.attn

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        mixed_query_layer = self.query(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)

        past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        if self.save_attention:
            self.save_attn(attention_probs)
        if self.save_gradient:
            attention_probs.register_hook(self.save_attn_gradients)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs_dropped = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs_dropped = attention_probs_dropped * head_mask

        context_layer = torch.matmul(attention_probs_dropped, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        outputs = outputs + (past_key_value,)
        return outputs

    
    
class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


class CustomBlip2QFormerModel(Blip2QFormerModel):
    def __init__(self, config: Blip2QFormerConfig):
        super().__init__(config)
        
        for layer_idx, layer in enumerate(self.encoder.layer):
            layer.attention.attention = CustomBlip2QFormerMultiHeadAttention(config)
            
            if layer_idx % config.cross_attention_frequency == 0:
                layer.crossattention.attention = CustomBlip2QFormerMultiHeadAttention(config, is_cross_attention=True)
                layer.has_cross_attention = True
            else:
                layer.has_cross_attention = False
            

class Blip2Model(Blip2PreTrainedModel):
    def __init__(self, config: Blip2Config):
        super().__init__(config)

        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))
        self.qformer = CustomBlip2QFormerModel(config.qformer_config)

        # convert visual dim 1024 from vit-large to 1048 for qformer input involving cross-attention
        self.qformer_input_proj = nn.Linear(config.mm_hidden_size, config.qformer_config.encoder_hidden_size)
        # self.proj = nn.Linear(config.mm_hidden_size, config.hidden_size)
        # modules = [nn.Linear(config.qformer_config.hidden_size, config.hidden_size), nn.GELU(), nn.Linear(config.hidden_size, config.hidden_size)]
        # self.proj = nn.Sequential(*modules)
        self.proj = nn.Linear(config.qformer_config.hidden_size, config.hidden_size)

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Returns:
            vision_outputs (`BaseModelOutputWithPooling` or tuple of `torch.FloatTensor`):
                The vision model outputs. If `return_dict=True`, the output is a [`BaseModelOutputWithPooling`] that
                contains the image features, the pooled image features and the hidden states if
                `output_hidden_states=True`.
        Examples:
        ```python
        >>> import torch
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import Blip2Processor, Blip2Model

        >>> device = "cuda" if torch.cuda.is_available() else "cpu"

        >>> processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        >>> model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
        >>> model.to(device)  # doctest: +IGNORE_RESULT

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
        >>> qformer_outputs = model.get_qformer_features(**inputs)
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # vision_outputs = self.vision_model(
        #     pixel_values=pixel_values,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )
        #
        # image_embeds = vision_outputs[0]
        
        # pixel_values: torch.Size([32, 576, 1024])
        # image_embeds: torch.Size([32, 576, 1408])
        image_embeds = self.qformer_input_proj(pixel_values)
        # image_embeds = pixel_values


        # print('pixel_values to proj', pixel_values.shape, image_embeds.shape)
        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        ).last_hidden_state
        # print('qformer out', query_outputs.shape)
        query_outputs = self.proj(query_outputs)
        return query_outputs

def qformer_config_template(config, projector_type):
    pattern = r"qformer(\d+)_(\d+)"

    match = re.search(pattern, projector_type)
    num_hidden_layers = int(match.group(1))
    num_query_tokens = int(match.group(2))

    qformer_config = type('Blip2Config', (PretrainedConfig,), {
        "initializer_factor": 1.0,
        "initializer_range": 0.02,
        "model_type": "blip-2",
        "num_query_tokens": num_query_tokens,
        "hidden_size": config.hidden_size,
        "mm_hidden_size": config.mm_hidden_size,
        "qformer_config": type('qformer_config', (PretrainedConfig,), {
            "_name_or_path": "",
            "add_cross_attention": False,
            "architectures": None,
            "attention_probs_dropout_prob": 0.0,
            "bad_words_ids": None,
            "begin_suppress_tokens": None,
            "bos_token_id": None,
            "chunk_size_feed_forward": 0,
            "classifier_dropout": None,
            "cross_attention_frequency": 2,
            "cross_attention_hidden_size": None,
            "decoder_start_token_id": None,
            "diversity_penalty": 0.0,
            "do_sample": False,
            "early_stopping": False,
            "encoder_hidden_size": 1408,
            "encoder_no_repeat_ngram_size": 0,
            "eos_token_id": None,
            "exponential_decay_length_penalty": None,
            "finetuning_task": None,
            "forced_bos_token_id": None,
            "forced_eos_token_id": None,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.0,
            "hidden_size": 768,
            "id2label": {
                "0": "LABEL_0",
                "1": "LABEL_1"
            },
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "is_decoder": False,
            "is_encoder_decoder": False,
            "label2id": {
                "LABEL_0": 0,
                "LABEL_1": 1
            },
            "layer_norm_eps": 1e-12,
            "length_penalty": 1.0,
            "max_length": 20,
            "max_position_embeddings": 512,
            "min_length": 0,
            "model_type": "blip_2_qformer",
            "no_repeat_ngram_size": 0,
            "num_attention_heads": 32,
            "num_beam_groups": 1,
            "num_beams": 1,
            "num_hidden_layers": num_hidden_layers,
            "num_return_sequences": 1,
            "output_attentions": False,
            "output_hidden_states": False,
            "output_scores": False,
            "pad_token_id": 0,
            "position_embedding_type": "absolute",
            "prefix": None,
            "problem_type": None,
            "pruned_heads": {},
            "remove_invalid_values": False,
            "repetition_penalty": 1.0,
            "return_dict": True,
            "return_dict_in_generate": False,
            "sep_token_id": None,
            "suppress_tokens": None,
            "task_specific_params": None,
            "temperature": 1.0,
            "tf_legacy_loss": False,
            "tie_encoder_decoder": False,
            "tie_word_embeddings": True,
            "tokenizer_class": None,
            "top_k": 50,
            "top_p": 1.0,
            "torch_dtype": None,
            "torchscript": False,
            "transformers_version": "4.27.0.dev0",
            "typical_p": 1.0,
            "use_bfloat16": False,
            "vocab_size": 30522
        })()
    })()
    return qformer_config




def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    elif projector_type == 'identity':
        return IdentityMap()

    elif projector_type.startswith('qformer'):  # qformer2_64
        qformer_config = qformer_config_template(config, projector_type)
        return Blip2Model(qformer_config)
    elif projector_type.startswith('cabstractor'):  # cabstractor_64 or cabstractor_144
        # C-Abstractor referring to https://github.com/kakaobrain/honeybee
        pattern = r"cabstractor_(\d+)"
        match = re.search(pattern, projector_type)
        num_query_tokens = int(match.group(1))
        print(f"For C-Abastractor: {num_query_tokens = }")
        return CAbstractorProjector(
                vision_dim=1024,
                llm_dim=4096,
                num_query_tokens=num_query_tokens,
                depth=3,  # the layer num of each ResNet Block which is same to the honeybee paper.
                mlp_depth=2,  # the layer num of the MLP Block that mapping projector dim to llm dim. It is same to the honeybee paper.
            )
    elif projector_type.startswith('dabstractor'): 
        # D-Abstractor referring to https://github.com/kakaobrain/honeybee
        pattern = r"dabstractor(\d+)_(\d+)"
        match = re.search(pattern, projector_type)
        num_layers = int(match.group(1))
        num_query_tokens = int(match.group(2))
        print(f"For D-Abastractor: {num_query_tokens = }")
        return DAbstractorProjector(
                vision_dim=1024,
                llm_dim=4096,
                num_query_tokens=num_query_tokens,
                layer_num=num_layers,  # the layer num of each ResNet Block which is same to the honeybee paper.
            )
    elif 'avgpool' in projector_type:
        match = re.match(r'avgpool(\d+)_(\d+)', projector_type)
        num_layers = int(match.group(1))
        num_query_tokens = int(match.group(2))
    
        print("For AvgPool:{} layer with {} query tokens".format(num_layers, num_query_tokens))
        
        return AvgPoolProjector(layer_num=num_layers, query_num=num_query_tokens, mm_hidden_size=config.mm_hidden_size, llm_hidden_size=config.hidden_size)
    
    elif 'maxpool' in projector_type:
        match = re.match(r'maxpool(\d+)_(\d+)', projector_type)
        num_layers = int(match.group(1))
        num_query_tokens = int(match.group(2))
    
        print("For MaxPool: {} layer with {} query tokens".format(num_layers, num_query_tokens))
        
        return MaxPoolProjector(layer_num=num_layers, query_num=num_query_tokens, mm_hidden_size=config.mm_hidden_size, llm_hidden_size=config.hidden_size)
    else:
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.hidden_size, config.hidden_size))
            return nn.Sequential(*modules)
    raise ValueError(f'Unknown projector type: {projector_type}')