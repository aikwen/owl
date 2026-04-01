from enum import Enum

class ExecMode(str, Enum):
    TRAIN = "train"
    VALIDATE = "validate"
    VISUALIZE = "visualization"

class EngineState(str, Enum):
    """整个 OwlEngine 的生命周期状态
    """
    EMPTY = "engine_empty"      # 仅初始化，无组件
    PENDING = "engine_pending"  # 组件已注入
    INITED = "engine_inited"    # 分配 Device，加载权重
    RUNNING = "engine_running"
    FINISHED = "engine_finished"
    ERROR = "engine_error"


class ExecState(str, Enum):
    """定义在特定 Mode 下的任务流转与 Epoch 调度。
    """
    EXEC_STARTED = "exec_started"
    EPOCH_STARTED = "epoch_started"
    MODE_TRAINING = "mode_training"
    MODE_VALIDATING = "mode_validating"
    MODE_VISUALIZING = "mode_visualizing"
    EPOCH_SAVING = "epoch_saving"
    EPOCH_ENDED = "epoch_ended"
    EXEC_ENDED = "exec_ended"


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