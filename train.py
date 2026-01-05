import os
import torch
from pprint import pprint
from configs.config import parser
from dataset.data_module import DataModule
from lightning_tools.callbacks import add_callbacks
# 根据use_causal_prior参数选择使用哪个模型
try:
    from models.R2GenGPT_causal import R2GenGPTCausal
except ImportError:
    R2GenGPTCausal = None
from models.R2GenGPT import R2GenGPT
from lightning.pytorch import seed_everything
import lightning.pytorch as pl

 
def train(args):
    dm = DataModule(args)
    callbacks = add_callbacks(args)

    # 如果使用DDP策略，设置find_unused_parameters=True以处理未使用的参数
    # 单GPU时使用auto策略，多GPU时使用DDP
    strategy = args.strategy
    if args.devices == 1:
        # 单GPU时使用auto策略
        strategy = "auto"
    elif strategy == "ddp" or (isinstance(strategy, str) and "ddp" in strategy.lower()):
        from lightning.pytorch.strategies import DDPStrategy
        strategy = DDPStrategy(find_unused_parameters=True)

    trainer = pl.Trainer(
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=strategy,
        accelerator=args.accelerator,
        precision=args.precision,
        val_check_interval = args.val_check_interval,
        limit_val_batches = args.limit_val_batches,
        max_epochs = args.max_epochs,
        num_sanity_val_steps = args.num_sanity_val_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks["callbacks"], 
        logger=callbacks["loggers"]
    )

    # 根据use_causal_prior或use_causal_vision参数选择使用哪个模型
    # 如果启用了文本因果先验或视觉因果先验，都需要使用R2GenGPTCausal模型
    use_causal_prior = getattr(args, 'use_causal_prior', False)
    use_causal_vision = getattr(args, 'use_causal_vision', False)
    use_causal = use_causal_prior or use_causal_vision
    
    if args.ckpt_file is not None:
        # 从checkpoint加载
        # 注意：由于checkpoint是自定义格式（不是标准Lightning格式），需要特殊处理
        try:
            checkpoint = torch.load(args.ckpt_file, map_location='cpu')
            print(f"加载checkpoint: {args.ckpt_file}")
            print(f"Checkpoint包含的键: {checkpoint.keys()}")
            
            # 检查checkpoint中是否有config信息
            if 'config' in checkpoint:
                # 使用checkpoint中的config更新args
                ckpt_config = checkpoint['config']
                for key, value in ckpt_config.items():
                    if not hasattr(args, key) or getattr(args, key) is None:
                        setattr(args, key, value)
                print("使用checkpoint中的配置参数")
            
            # 创建模型实例
            if use_causal and R2GenGPTCausal is not None:
                try:
                    model = R2GenGPTCausal(args)
                    print("创建因果先验模型实例")
                except Exception as e:
                    print(f"警告: 无法创建因果先验模型: {e}")
                    print("尝试创建原始模型")
                    model = R2GenGPT(args)
            else:
                model = R2GenGPT(args)
            
            # 加载模型权重
            if 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'], strict=False)
                print("成功加载模型权重")
                if 'epoch' in checkpoint:
                    print(f"Checkpoint来自 epoch {checkpoint['epoch']}, step {checkpoint.get('step', 'unknown')}")
            else:
                print("警告: checkpoint中没有找到'model'键，尝试直接加载")
                model.load_state_dict(checkpoint, strict=False)
                
        except Exception as e:
            print(f"加载checkpoint时出错: {e}")
            print("尝试使用Lightning标准方式加载...")
            # Fallback: 尝试使用Lightning标准方式
            if use_causal and R2GenGPTCausal is not None:
                try:
                    model = R2GenGPTCausal.load_from_checkpoint(args.ckpt_file, strict=False)
                    print("从checkpoint加载因果先验模型（Lightning方式）")
                except:
                    print("警告: 无法加载因果先验模型，尝试加载原始模型")
                    model = R2GenGPT.load_from_checkpoint(args.ckpt_file, strict=False)
            else:
                model = R2GenGPT.load_from_checkpoint(args.ckpt_file, strict=False)
    else:
        # 创建新模型
        if use_causal and R2GenGPTCausal is not None:
            model = R2GenGPTCausal(args)
            print("使用因果先验模型 (R2GenGPTCausal)")
        else:
            model = R2GenGPT(args)
            if use_causal:
                print("警告: 因果先验已启用但无法加载因果模型，使用原始模型")
            else:
                print("使用原始模型 (R2GenGPT)")

    if args.test:
        trainer.test(model, datamodule=dm)
    elif args.validate:
        trainer.validate(model, datamodule=dm)
    else:
        trainer.fit(model, datamodule=dm)

def main():
    args = parser.parse_args()
    os.makedirs(args.savedmodel_path, exist_ok=True)
    pprint(vars(args))
    seed_everything(7508, workers=True)
    train(args)


if __name__ == '__main__':
    main()