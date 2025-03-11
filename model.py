import math
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


def new_gelu(x):
    """
    实现 Google BERT 仓库中的 GELU 激活函数（与 OpenAI GPT 中的实现相同）。

    参数:
        x (torch.Tensor): 输入张量。

    返回:
        torch.Tensor: 应用 GELU 激活函数后的张量。
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class LayerNorm(nn.Module):
    """
    LayerNorm 层，但带有可选的偏置。PyTorch 的 LayerNorm 默认不支持 bias=False。

    参数:
        ndim (int): 归一化的维度。
        bias (bool): 是否使用偏置，默认为 True。如果为 False，则不使用偏置。
    """
    def __init__(self, ndim, bias):
        super().__init__()
        # 初始化权重参数
        self.weight = nn.Parameter(torch.ones(ndim))
        # 如果使用偏置，则初始化偏置参数；否则为 None
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        """
        前向传播函数。

        参数:
            input (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 归一化后的张量。
        """
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """
    因果自注意力机制模块，用于处理序列数据，确保每个位置只能关注其之前的位置。

    参数:
        config: 配置参数，包含以下属性：
            n_embd (int): 嵌入维度。
            n_head (int): 多头注意力机制中的头数。
            bias (bool): 是否使用偏置，默认为 True。
            dropout (float): Dropout 概率。
            block_size (int): 块大小，用于创建因果掩码。
            is_train (bool): 是否处于训练模式。
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # 为所有头计算键、查询和值投影，但以批处理方式
        # 键、查询和值线性变换层
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        # 输出投影
        # 输出线性变换层
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # 正则化
        # 注意力 Dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        # 残差 Dropout
        self.resid_dropout = nn.Dropout(config.dropout)
        # 头数
        self.n_head = config.n_head
        # 嵌入维度
        self.n_embd = config.n_embd
        # Dropout 概率
        self.dropout = config.dropout
        # 是否处于训练模式
        self.is_train=config.is_train

        # Flash Attention 可以让 GPU 运行更快，但仅在 PyTorch >= 2.0 中受支持
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        # 检查是否支持 Flash Attention
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # 因果掩码，确保注意力仅应用于输入序列的左侧
            # 注册因果掩码缓冲区
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, kvcache=None):
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, sequence_length, embedding_dim)。
            kvcache (Tuple[torch.Tensor, torch.Tensor], optional): 键和值缓存，默认为 None。

        返回:
            Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]: 返回注意力机制的输出和更新后的键值缓存。
        """
        # 获取批量大小、序列长度和嵌入维度
        B, T, C = x.size() 

        # 计算所有头的查询、键和值，并将头维度移到批量维度前
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2) # 分割为查询、键和值
        
        if kvcache:
            # 获取缓存的键和值
            prev_k, prev_v = kvcache
            # 拼接缓存的键和当前的键
            k = torch.cat([prev_k, k], dim=1)
            # 拼接缓存的值和当前的值
            v = torch.cat([prev_v, v], dim=1)

        # 更新键值缓存
        new_kvcache = [k, v]
        # 获取当前序列长度
        curr_T = k.shape[1]

        # 重塑张量形状为 (B, nh, T, hs)
        k = k.view(B, curr_T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, curr_T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # 因果自注意力；自注意力计算: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # 使用 Flash Attention CUDA 内核进行高效注意力计算
            # 使用因果掩码进行慢速注意力计算
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=self.is_train)
        
        # 重新组装所有头的输出
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # 输出投影
        y = self.resid_dropout(self.c_proj(y))

        # 返回输出和更新后的键值缓存
        return y, new_kvcache


class MLP(nn.Module):
    """
    MLP（多层感知机）模块，用于在 Transformer 块中执行前馈神经网络操作。

    参数:
        config: 配置参数，包含以下属性：
            n_embd (int): 嵌入维度。
            bias (bool): 是否使用偏置，默认为 True。
            dropout (float): Dropout 概率。
    """
    def __init__(self, config):
        super().__init__()
        # 第一个线性层，扩展维度
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        # 第二个线性层，缩减维度
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        # Dropout 层
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: MLP 层的输出。
        """
        # 第一个线性变换
        x = self.c_fc(x)
        # 应用 GELU 激活函数
        x = new_gelu(x)
        # 第二个线性变换
        x = self.c_proj(x)
        # 应用 Dropout
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    Transformer 块，包含层归一化、因果自注意力机制、另一个层归一化和 MLP 层。

    参数:
        config: 配置参数，包含以下属性：
            n_embd (int): 嵌入维度。
            bias (bool): 是否使用偏置，默认为 True。
            dropout (float): Dropout 概率。
            n_head (int): 多头注意力机制中的头数。
            其他参数如 block_size 等也包含在 config 中。
    """
    def __init__(self, config):
        super().__init__()
        # 第一个层归一化
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        # 因果自注意力机制
        self.attn = CausalSelfAttention(config)
        # 第二个层归一化
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        # MLP 层
        self.mlp = MLP(config)

    def forward(self, x, kvcache=None):
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, sequence_length, embedding_dim)。
            kvcache (List[torch.Tensor], optional): 键值缓存列表，默认为 None。

        返回:
            Tuple[torch.Tensor, List[torch.Tensor]]: 返回 Transformer 块的输出和更新后的键值缓存。
        """
        # 应用层归一化和因果自注意力机制
        attn_out, cache_ele = self.attn(self.ln_1(x), kvcache)
        # 残差连接
        x = x + attn_out
        # 应用层归一化、MLP 层和残差连接
        x = x + self.mlp(self.ln_2(x))
        # 返回输出和更新后的键值缓存
        return x, cache_ele


# 定义 GPT 配置参数
@dataclass
class GPTConfig:
    """
    GPT 配置参数类，用于存储 GPT 模型的配置参数。

    参数:
        block_size (int): 块大小，默认为 1024。
        vocab_size (int): 词汇表大小，默认为 50304（GPT-2 的词汇表大小为 50257，填充到 64 的倍数以提高效率）。
        n_layer (int): Transformer 层的数量，默认为 12。
        n_head (int): 多头注意力机制中的头数，默认为 12。
        n_embd (int): 嵌入维度，默认为 768。
        dropout (float): Dropout 概率，默认为 0.0。
        bias (bool): 是否使用偏置，默认为 True（与 GPT-2 相同）。如果为 False，则性能略好且速度更快。
        is_train (bool): 是否处于训练模式，默认为 True。
    """
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 的词汇表大小为 50257，填充到 64 的倍数以提高效率
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: 与 GPT-2 相同的偏置；False: 性能略好且速度更快
    is_train :bool = True


class GPT(nn.Module):
    """
    GPT（生成式预训练 Transformer）模型类。

    参数:
        config: GPT 配置参数。
    """
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # 定义 Transformer 模块字典
        self.transformer = nn.ModuleDict(dict(
            wpe = nn.Embedding(config.block_size, config.n_embd), # 位置嵌入
            drop = nn.Dropout(config.dropout), # Dropout 层
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # Transformer 层列表
            ln_f = LayerNorm(config.n_embd, bias=config.bias), # 最终层归一化
        ))

        # 语言模型头（线性层）
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) 
        
        # 初始化所有权重
        self.apply(self._init_weights)
        # 对残差投影应用特殊的 sqrt初始化，根据 GPT-2 论文
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # 报告参数数量
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        返回模型中的参数数量。
        对于非嵌入计数（默认），位置嵌入会被减去。
        令牌嵌入也会被减去，但由于参数共享，这些参数实际上被用作最终层的权重，因此我们包括它们。

        参数:
            non_embedding (bool): 是否排除嵌入参数，默认为 True。

        返回:
            int: 参数数量。
        """
        # 计算总参数数量
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            # 减去位置嵌入参数数量
            n_params -= self.transformer.wpe.weight.numel()
        # 返回参数数量
        return n_params

    def _init_weights(self, module):
        """
        初始化权重。

        参数:
            module (nn.Module): 要初始化的模块。
        """
        if isinstance(module, nn.Linear):
            # 初始化线性层权重
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                # 初始化线性层偏置
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # 初始化嵌入层权重
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, emb, targets=None, kvcache=None):
        """
        前向传播函数。

        参数:
            emb (torch.Tensor): 输入嵌入张量，形状为 (batch_size, sequence_length, embedding_dim)。
            targets (torch.Tensor, optional): 目标张量，形状为 (batch_size, sequence_length)，默认为 None。
            kvcache (List[torch.Tensor], optional): 键值缓存列表，默认为 None。

        返回:
            Tuple[torch.Tensor, Optional[torch.Tensor], List[torch.Tensor]]: 返回 logits、损失和更新后的键值缓存。
        """
        device = emb.device
        # 获取批量大小和序列长度
        b,t,_ = emb.size()
        # 检查序列长度是否超过块大小
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        # 生成位置张量，形状为 (1, t)
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # 生成位置嵌入，形状为 (1, t, n_embd)
        pos_emb = self.transformer.wpe(pos) 
        # 应用 Dropout 并添加位置嵌入
        x = self.transformer.drop(emb + pos_emb)

        if not kvcache:
            # 如果没有提供缓存，则初始化缓存列表
            kvcache = [None] * self.config.n_layer
        else:
            # 如果提供了缓存，则仅使用最后一个时间步的嵌入
            x = x[:, [-1], :]

        # 初始化新的缓存列表
        new_kvcache = []
        for block, kvcache_block in zip(self.transformer.h, kvcache):
            # 应用 Transformer 块
            x, cache_ele = block(x, kvcache=kvcache_block)
            # 添加缓存元素到新的缓存列表
            new_kvcache.append(cache_ele)

        # 应用最终层归一化
        x = self.transformer.ln_f(x)

        # 如果提供了目标，则计算损失
        if targets is not None:
            # 计算 logits
            logits = self.lm_head(x) 
            # 计算交叉熵损失
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1000)
        else:
            # 如果没有提供目标，则仅计算最后一个时间步的 logits
            # 计算最后一个时间步的 logits
            logits = self.lm_head(x[:, [-1], :]) 
            # 损失为 None
            loss = None

        # 返回 logits、损失和新的缓存列表
        return logits, loss, new_kvcache

    def crop_block_size(self, block_size):
        """
        如果需要，减少块大小。
        例如，我们可能加载 GPT2 预训练模型检查点（块大小 1024），但希望使用更小的块大小用于某些更小、更简单的模型。

        参数:
            block_size (int): 新的块大小。
        """
        # 检查新的块大小是否小于或等于当前块大小
        assert block_size <= self.config.block_size
        # 更新配置中的块大小
        self.config.block_size = block_size
        # 截断位置嵌入权重
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                # 截断注意力偏置
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        """
        从预训练模型加载权重。

        参数:
            model_type (str): 模型类型，可以是 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'。
            override_args (dict, optional): 要覆盖的配置参数，默认为 None。

        返回:
            GPT: 加载了预训练权重的 GPT 实例。
        """
        # 检查模型类型是否有效
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        # 默认覆盖参数为空字典
        override_args = override_args or {} 
        # 检查是否只覆盖 dropout 参数
        assert all(k == 'dropout' for k in override_args)
        # 从 transformers 库导入 GPT2LMHeadModel
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # 根据模型类型确定配置参数
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type] # 获取配置参数
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # 词汇表大小，GPT 模型检查点始终为 50257
        config_args['block_size'] = 1024 # 块大小，GPT 模型检查点始终为 1024
        config_args['bias'] = True # 偏置，GPT 模型检查点始终为 True
        # 我们可以覆盖 dropout 率，如果需要
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']

        # 创建从头初始化的 minGPT 模型
        # 创建配置对象
        config = GPTConfig(**config_args)
        # 创建模型实例
        model = GPT(config)
        # 获取模型状态字典
        sd = model.state_dict()
        # 获取状态字典的键
        sd_keys = sd.keys()
        # 排除注意力偏置键，因为它们不是参数
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] 

        # 初始化 Huggingface/transformers 模型
        model_hf = GPT2LMHeadModel.from_pretrained(model_type) # 加载预训练模型
        # 获取预训练模型状态字典
        sd_hf = model_hf.state_dict()

        # 复制权重，同时确保所有参数对齐，名称和形状匹配
        # 获取预训练模型状态字典的键
        sd_keys_hf = sd_hf.keys()
        # 排除掩码偏置键，因为它们只是缓冲区
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] 
        # 排除注意力偏置键，因为它们只是掩码（缓冲区）
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] 
        # 需要转置的权重键
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # 基本上 OpenAI 的检查点使用 "Conv1D" 模块，但我们只想要使用普通的线性层
        # 这意味着我们必须在导入时转置这些权重
        # 检查键的数量是否匹配
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # 特殊处理需要转置的 Conv1D 权重
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    # 转置复制权重
                    sd[k].copy_(sd_hf[k].t())
            else:
                # 直接复制其他参数
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    # 直接复制权重
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        配置优化器。

        参数:
            weight_decay (float): 权重衰减参数。
            learning_rate (float): 学习率。
            betas (Tuple[float, float]): AdamW 优化器的 beta 参数。
            device_type (str): 设备类型。

        返回:
            optim.Optimizer: 配置好的优化器。
        """
        # 获取所有候选参数
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # 过滤掉不需要梯度的参数
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # 创建优化器组。任何 2D 参数将进行权重衰减，否则不进行权重衰减。
        # 即，所有矩阵乘法和嵌入中的权重张量衰减，所有偏置和层归一化不进行衰减。
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2] # 衰减参数列表
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2] # 不衰减参数列表
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay}, # 衰减参数组
            {'params': nodecay_params, 'weight_decay': 0.0} # 不衰减参数组
        ]
        # 计算衰减参数数量
        num_decay_params = sum(p.numel() for p in decay_params)
        # 计算不衰减参数数量
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        
        # 创建 AdamW 优化器，并使用融合版本如果可用
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        # 如果支持且设备为 CUDA，则使用融合版本
        use_fused = fused_available and device_type == 'cuda'
        # 设置融合参数
        extra_args = dict(fused=True) if use_fused else dict()
        # 创建优化器
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas,eps=1e-5,**extra_args)
        print(f"using fused AdamW: {use_fused}")

        # 返回优化器
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """
        估计模型每秒浮点运算利用率（MFU），以 A100 bfloat16 峰值 FLOPS 为单位。

        参数:
            fwdbwd_per_iter (int): 每次迭代的前后向传播次数。
            dt (float): 每次迭代的时间。

        返回:
            float: 估计的 MFU。
        """
        # 首先估计每次迭代的浮点运算次数。
        # 获取参数数量
        N = self.get_num_params()
        # 获取配置参数
        cfg = self.config
        # 获取层数、头数、每个头的维度、块大小
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        # 计算每个 token 的浮点运算次数
        flops_per_token = 6*N + 12*L*H*Q*T
        # 计算每次前后向传播的浮点运算次数
        flops_per_fwdbwd = flops_per_token * T
        # 计算每次迭代的浮点运算次数
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # 将我们的浮点运算吞吐量表示为 A100 bfloat16 峰值 FLOPS 的比率
        flops_achieved = flops_per_iter * (1.0/dt) # 每秒的浮点运算次数
        # A100 GPU bfloat16 峰值 FLOPS 为 312 TFLOPS
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        # 计算 MFU
        mfu = flops_achieved / flops_promised 
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        生成文本序列。

        参数:
            idx (torch.Tensor): 输入索引张量，形状为 (batch_size, sequence_length)。
            max_new_tokens (int): 要生成的新的 token 数量。
            temperature (float): 温度参数，默认为1.0。
            top_k (int, optional): 最高的 k 个概率，默认为 None。

        返回:
            torch.Tensor: 生成的索引张量，形状为 (batch_size, sequence_length + max_new_tokens)。
        """
        # 初始化键值缓存
        kvcache = None
        for _ in range(max_new_tokens):
            # 如果序列上下文增长过长，则必须在块大小处裁剪
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:] # 裁剪索引
            # 前向传播模型以获取序列中索引的 logits
            logits, _, kvcache = self(idx_cond, kvcache=kvcache) # 前向传播
            # 获取最后一个时间步的 logits 并按所需温度缩放
            logits = logits[:, -1, :] / temperature # 缩放 logits
            # 可选地，将 logits 裁剪到仅包含前 k 个选项
            if top_k is not None:
                # 获取前 k 个值
                v, _ = torch.topk(logits, min(top_k, logits.size(-1))) 
                # 将小于第 k 大的值设为 -inf
                logits[logits < v[:, [-1]]] = -float('Inf')
            # 应用 softmax 将 logits 转换为（归一化）概率
            probs = F.softmax(logits, dim=-1) # 计算 softmax
            # 从分布中采样
            idx_next = torch.multinomial(probs, num_samples=1) # 采样下一个索引
            # 将采样的索引追加到运行序列并继续
            idx = torch.cat((idx, idx_next), dim=1) # 拼接索引

        # 返回生成的索引
        return idx

