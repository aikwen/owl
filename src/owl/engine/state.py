from enum import Enum

class ExecMode(str, Enum):
    TRAIN = "train"
    VALIDATE = "validate"
    VISUALIZE = "visualization"

class AppState(str, Enum):
    """整个 OwlEngine 的生命周期状态
    """
    EMPTY = "app_empty"      # 仅初始化，无组件
    INSTANTIATED = "app_instantiated"  # 初始化组件
    MOUNTED = "app_mounted"    # 分配 Device，加载权重
    RUNNING = "app_running"
    FINISHED = "app_finished"
    ERROR = "app_error"


class ExecState(str, Enum):
    """完全对应多态路由图的节点"""
    START = "start"
    ROUTING = "routing"  # 路由分支
    TRAIN = "train"
    VALIDATE = "validate"
    VISUAL = "visual"
    END = "end"          # 结束


class StepState(str, Enum):
    """定义 StepPipeline 处理单一 Batch 时的状态。
    """
    STARTED = "step_started"                      # 拉取数据至 Device
    GRAD_ZEROED = "step_grad_zeroed"              # 梯度清空
    FORWARD_COMPUTED = "step_forward_computed"    # 前向传播，获得 outputs
    LOSS_COMPUTED = "step_loss_computed"          # 损失计算，获得 loss 标量
    BACKWARD_COMPUTED = "step_backward_computed"  # 反向传播：计算梯度
    OPTIMIZED = "step_optimized"                  # 权重更新
    SCHEDULED = "step_scheduled"                  # 学习率更新
    ENDED = "step_ended"