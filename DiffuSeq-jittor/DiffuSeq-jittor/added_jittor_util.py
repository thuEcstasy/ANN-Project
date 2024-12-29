import jittor as jt
import math

def clip_grad_norm_(parameters, max_grad_norm):
    # 获取所有参数的梯度
    parameters = [p for p in parameters if p.grad is not None]
    total_norm = 0.0

    # 计算梯度的总范数（2 范数）
    for p in parameters:
        param_norm = jt.norm(p.grad, p=2)  # 计算每个参数的梯度范数
        total_norm += param_norm ** 2
    total_norm = math.sqrt(total_norm)  # 求总范数

    # 如果总范数超过 max_grad_norm，按比例缩放梯度
    clip_coef = max_grad_norm / (total_norm + 1e-6)  # 避免除零
    if clip_coef < 1:
        for p in parameters:
            p.grad *= clip_coef  # 缩放梯度

    return total_norm
