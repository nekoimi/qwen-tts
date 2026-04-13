#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# nekoimi 2026/4/13


if __name__ == '__main__':
    import torch
    import sys

    print(f"Python 路径: {sys.executable}")
    print(f"Torch 版本: {torch.__version__}")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        if "+cpu" in torch.__version__:
            print(">>> 结论：你安装的依然是 CPU 版本，请检查是否在正确的 .venv 下运行了 pip。")
        else:
            print(">>> 结论：版本是对的，但 Torch 找不到你的显卡驱动，尝试重启电脑。")
    else:
        print(f">>> 成功！检测到显卡: {torch.cuda.get_device_name(0)}")
