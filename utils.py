import os
import math
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group
from contextlib import nullcontext
import transformers
from typing import Dict, Optional, Sequence, List

from data import get_batch


def setup_environment(config):
    """
    设置训练环境，包括分布式数据并行（DDP）如果适用的话。

    参数:
        config (dict): 包含训练配置参数的字典。预期的键包括：
            - 'backend' (str): 分布式后端，例如 'nccl' 或 'gloo'。
            - 'device' (str): 设备类型，例如 'cuda:0' 或 'cpu'。
            - 'gradient_accumulation_steps' (int): 梯度累积步数。
            - 'batch_size' (int): 每个批次的样本数量。
            - 'block_size' (int): 输入序列的长度。
            - 'out_dir' (str): 输出目录，用于保存日志和检查点。
            - 'dtype' (str): 数据类型，例如 'float32', 'bfloat16', 'float16'。
    """
    # 检查是否运行在分布式数据并行（DDP）模式下
    # WORLD_SIZE 环境变量表示全局进程数，如果大于1，则启用 DDP
    ddp = int(os.environ.get('WORLD_SIZE', 1)) > 1
    
    if ddp:
        """
        分布式数据并行（DDP）模式下的初始化
        """
        # 初始化进程组，指定分布式后端
        init_process_group(backend=config['backend'])

        # 从环境变量中获取当前进程的排名和本地排名
        ddp_rank = int(os.environ['RANK']) # 全局排名
        ddp_local_rank = int(os.environ['LOCAL_RANK']) # 本地排名（GPU 编号）
        ddp_world_size = int(os.environ['WORLD_SIZE']) # 进程总数

        # 设置当前进程使用的 CUDA 设备
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)

        # 确定是否为 master 进程（rank 0），用于日志记录和保存检查点
        master_process = ddp_rank == 0  

        # 为每个进程设置不同的随机种子偏移量，避免不同进程产生相同的随机数
        seed_offset = ddp_rank  
        
        # 调整梯度累积步数以适应 DDP 设置
        # 确保梯度累积步数可以被 WORLD_SIZE 整除
        gradient_accumulation_steps = config['gradient_accumulation_steps']
        assert gradient_accumulation_steps % ddp_world_size == 0
        # 每个进程的实际梯度累积步数
        gradient_accumulation_steps //= ddp_world_size
    else:
        # 单个进程即为 master
        master_process = True
        # 无需偏移随机种子
        seed_offset = 0
        # 单个进程
        ddp_world_size = 1
        # 本地排名为 0
        ddp_local_rank = 0
        # 使用配置中指定的设备
        device = config['device']
        # 使用配置中指定的梯度累积步数
        gradient_accumulation_steps = config['gradient_accumulation_steps']
    
    # 计算每次迭代处理的 token 数量，用于报告和监控
    tokens_per_iter = gradient_accumulation_steps * ddp_world_size * config['batch_size'] * config['block_size']
    print(f"tokens per iteration will be: {tokens_per_iter:,}")
    
    # 创建输出目录（仅在 master 进程中进行）
    if master_process:
        os.makedirs(config['out_dir'], exist_ok=True)
    
    # 设置随机种子，确保不同进程生成不同的随机数
    torch.manual_seed(1337 + seed_offset)
    
    # 在 A100 GPU 上启用 TF32 精度以加快训练速度
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # 确定设备类型和精度
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    # 根据配置中的 dtype 设置精度类型
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config['dtype']]
    # 根据设备类型决定是否使用自动混合精度（AMP）
    # 如果是 CPU，则不使用 AMP；否则，使用 AMP 并指定设备类型和精度
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    # 更新配置中的梯度累积步数
    config['gradient_accumulation_steps'] = gradient_accumulation_steps
    
    # 返回包含环境变量和配置参数的字典
    return {
        'device': device, # 当前设备，例如 'cuda:0'
        'device_type': device_type, # 设备类型，'cuda' 或 'cpu'
        'ctx': ctx, # 上下文管理器，用于自动混合精度
        'master_process': master_process, # 是否为 master 进程
        'ddp': ddp, # 是否启用 DDP
        'ddp_local_rank': ddp_local_rank, # 本地排名（GPU 编号）
        'ddp_world_size': ddp_world_size, # 进程总数
        'tokens_per_iter': tokens_per_iter, # 每次迭代处理的 token 数量
        'gradient_accumulation_steps': gradient_accumulation_steps # 实际的梯度累积步数
    }


@torch.no_grad()
def estimate_loss(model, train_dataloader,context_length,eval_iters, ctx, device, device_type, llm_model):
    """
    在评估迭代中估计损失。

    参数:
        model (torch.nn.Module): 需要评估的模型。
        train_dataloader (torch.utils.data.DataLoader): 训练数据加载器，用于获取批量数据。
        context_length (int): 输入序列的长度。
        eval_iters (int): 评估的迭代次数。
        ctx (contextlib.nullcontext 或 torch.amp.autocast): 上下文管理器，用于自动混合精度（如果适用）。
        device (str): 当前设备，例如 'cuda:0'。
        device_type (str): 设备类型，'cuda' 或 'cpu'。
        llm_model (Any): 大型语言模型相关的参数或实例，具体取决于实现。

    返回:
        dict: 一个包含每个数据拆分（此处为 'train'）的平均损失的字典。
    """
    # 用于存储结果的字典
    out = {}
    # 将模型设置为评估模式，禁用 dropout 等训练操作
    model.eval()
    for split in ['train']:
        # 初始化一个张量来存储每次迭代的损失
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            # 获取一个批次的输入数据 X 和目标数据 Y
            X, Y = get_batch(train_dataloader,context_length,device,device_type, llm_model)
            with ctx:
                try:
                    # 前向传播，获取 logits、损失值和其他可能的输出
                    logits, loss, _ = model(X, Y)
                except Exception as e:
                    print(f"Error during evaluation: {str(e)}")
                    continue
            # 将损失值转换为 Python 浮点数并存储
            losses[k] = loss.item()
        # 计算平均损失并存储在结果字典中
        out[split] = losses.mean()
    # 将模型设置回训练模式
    model.train()
    # 返回包含平均损失的结果字典
    return out


def get_lr(config, it):
    """
    根据学习率调度策略获取当前迭代的学习率。

    参数:
        config (dict): 包含学习率调度相关配置参数的字典。预期的键包括：
            - 'decay_lr' (bool): 是否启用学习率衰减。
            - 'learning_rate' (float): 初始学习率。
            - 'warmup_iters' (int): 学习率预热的迭代次数。
            - 'lr_decay_iters' (int): 学习率衰减的迭代次数。
            - 'min_lr' (float): 学习率的最小值。
        it (int): 当前迭代次数。

    返回:
        float: 当前迭代的学习率。
    """
    if not config['decay_lr']:
        # 如果不启用学习率衰减，返回初始学习率
        return config['learning_rate']
    
    # Linear warmup phase
    if it < config['warmup_iters']:
        # 线性预热阶段，学习率从 0 逐渐增加到初始学习率
        return config['learning_rate'] * it / config['warmup_iters']
    
    # After decay iterations, return min learning rate
    if it > config['lr_decay_iters']:
        # 超过衰减迭代次数后，返回最小学习率
        return config['min_lr']
    
    # 在预热和衰减之间的阶段，使用余弦衰减
    decay_ratio = (it - config['warmup_iters']) / (config['lr_decay_iters'] - config['warmup_iters'])
    assert 0 <= decay_ratio <= 1
    # 计算余弦衰减系数，范围从 1 到 0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    # 计算当前学习率
    return config['min_lr'] + coeff * (config['learning_rate'] - config['min_lr'])


def setup_wandb(config, master_process):
    """
    如果启用，则初始化 wandb 日志记录。

    参数:
        config (dict): 包含 wandb 配置参数的字典。预期的键包括：
            - 'wandb_log' (bool): 是否启用 wandb 日志记录。
            - 'wandb_project' (str): wandb 项目名称。
            - 'wandb_run_name' (str): wandb 运行名称。
        master_process (bool): 是否为 master 进程。

    返回:
        wandb.wandb_run.Run 或 None: wandb 运行实例或 None。
    """
    if config['wandb_log'] and master_process:
        try:
            import wandb
            print(f"Initializing wandb with project: {config['wandb_project']}, run name: {config['wandb_run_name']}")
            # 初始化 wandb
            wandb.init(project=config['wandb_project'], name=config['wandb_run_name'], config=config)
            return wandb
        except ImportError:
            print("wandb not installed, skipping wandb initialization")
            return None
    # 如果未启用 wandb 或不是 master 进程，返回 None
    return None


def save_checkpoint(config, model, optimizer, iter_num, model_args):
    """
    保存模型检查点。

    参数:
        config (dict): 包含保存检查点相关配置参数的字典。预期的键包括：
            - 'out_dir' (str): 输出目录路径。
            - 'checkpoint_filename' (str, 可选): 检查点文件名，默认为 'ckpt.pt'。
            - 'always_save_checkpoint' (bool): 是否始终保存检查点备份。
        model (torch.nn.Module): 需要保存的模型。
        optimizer (torch.optim.Optimizer): 优化器实例。
        iter_num (int): 当前迭代次数。
        model_args (Any): 模型的其他参数或配置，具体取决于实现。
    """
    # 如果模型是 DDP 实例，则获取原始模型
    raw_model = model.module if isinstance(model, DDP) else model
    
    # 构建检查点字典，包含模型状态、优化器状态、模型参数、当前迭代次数和配置
    checkpoint = {
        'model': raw_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'iter_num': iter_num,
        'config': config,
    }
    
    # 构建检查点文件路径
    checkpoint_path = os.path.join(config['out_dir'], config.get('checkpoint_filename', 'ckpt.pt'))
    print(f"Saving checkpoint to {checkpoint_path}")
    # 保存检查点到指定路径
    torch.save(checkpoint, checkpoint_path)
    
    # 如果配置中启用了始终保存检查点备份，则保存备份副本
    if config.get('always_save_checkpoint', False):
        backup_path = os.path.join(config['out_dir'], f'ckpt_{iter_num}.pt')
        print(f"Saving backup checkpoint to {backup_path}")
        torch.save(checkpoint, backup_path)


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """
    调整 Tokenizer 和嵌入层的大小。

    注意: 这是未优化的版本，可能会导致嵌入层大小不能被 64 整除。

    参数:
        special_tokens_dict (dict): 包含特殊 Token 的字典，例如：
            {
                'pad_token': '[PAD]',
                'bos_token': '[BOS]',
                'eos_token': '[EOS]',
                'additional_special_tokens': ['[SPECIAL1]', '[SPECIAL2]']
            }
        tokenizer (transformers.PreTrainedTokenizer): 预训练的 Tokenizer 实例。
        model (transformers.PreTrainedModel): 预训练的模型实例。
    """
    # 添加特殊 Token 到 Tokenizer 中，并获取新增的 Token 数量
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    # 调整模型的嵌入层大小以匹配新的 Tokenizer 大小
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        # 获取输入和输出嵌入层的权重数据
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        # 计算现有嵌入层的平均向量（不包括新添加的特殊 Token）
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        # 将新添加的特殊 Token 的嵌入向量初始化为平均向量
        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

