# DeCo

[Deco: Decoupling token compression from semanchc abstraction in multimodal large language models](https://arxiv.org/abs/2405.20985)



This study examines the projector module by interpreting the vision-language semantic flow within MLLMs. Our findings reveal that compressive projectors (e.g., QFormer), abstract visual patches into a limited set of semantic concepts, such as objects or attributes, resulting in a '*double abstraction*' phenomenon. The double abstraction is inefficient in training and will result in cumulative vision semantics deficiency. To mitigate this issue, we propose the key insight of **Decouple Compression from Abstraction (DeCo)**, *that is compressing the visual token number at the patch level by projectors and allowing the LLM to handle visual semantic abstraction entirely*. Consequently, we adopt a simple compressor, i.e., **2D Adaptive Pooling**, to downsample visual patches in a parameter-free manner. Empirical evaluation demonstrates that DeCo surpasses traditional compressive projectors regarding both performance and efficiency. It achieves performance gains of 0.9%, 7.1%, and 2.9% across the MLLM Benchmarks, Visual Localization, and Open-ended VQA tasks with fewer trainable parameters and faster convergence speed.

![image](https://github.com/yaolinli/DeCo/assets/24662157/84752529-f608-4d49-b686-1a5bff63a5f3)



We visualize the vision-language relevance maps across the same MLLM architecture except for projector modules in the following figure. The linear projector is non-compressive while the Q-Former and Adaptive Average Pooling (ours) compress the original 576 vision tokens to 64 tokens. Text-to-Patch relevance reveals the effective vision semantics aligned with the LLM during image-to-text generation. For Q-Former in the second row, its Query-to-Patch map discards the fine-grained visual semantics about “purple and red”. This semantic deficiency is transmitted to the final Text-to-Patch map and leads to a misalignment of vision patches and textual words.

![image](https://github.com/yaolinli/DeCo/assets/24662157/f443a5da-0d15-4fea-85e7-657ffe0a4d01)



# 2D Adaptive Pooling

Under the DeCo architecture, we employ the 2D Adaptive Average Pooling as a natural downsampler of the visual tokens at the patch level. Given N patch tokens from the ViT, the adaptive pooling can reduce the token number N to a lesser square number M.  These tokens are finally projected by the linear layer to match the textual embedding dimension, serving as visual inputs to the LLM.

The core code using 2D Adaptive Pooling as the projector is:

```
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
        batch_size, seq_len, h_dim = visual_feat.shape  # 576
        hw = int(seq_len ** 0.5)  # 24
        shaped_visual_feat = rearrange(visual_feat, "b (h w) d -> b d h w", h=hw, w=hw)  # torch.Size([64, 1024, 24, 24])
        pooled_visual_feat = self.sampler(shaped_visual_feat)  # torch.Size([64, 1024, 12, 12])
        reshaped_visual_feat = rearrange(pooled_visual_feat, "b d h w -> b (h w) d")  # [64, 144, 1024]
        output_feat = self.mlp_projector(reshaped_visual_feat)  # [64, 144, 4096])
        return output_feat
```



The complete code will come soon.
