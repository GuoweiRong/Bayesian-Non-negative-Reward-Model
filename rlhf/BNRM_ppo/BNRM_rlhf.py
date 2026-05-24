# 1. 先执行你的注册逻辑 —— 只要 import 这个模块，register_model 就会跑
from BNRM_register import *  # noqa: F401

# 2. 再调用原始的 rlhf_main
from swift.llm import rlhf_main

if __name__ == "__main__":
    # 这里直接把命令行参数交给 rlhf_main
    rlhf_main()
