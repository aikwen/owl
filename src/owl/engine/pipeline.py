from typing import Any

from statemachine import StateMachine, State
import torch

from .state import StepState
from ..toolkits.criterion.base import OwlCriterion
from ..toolkits.data.types import DataSetBatch
from ..toolkits.model.base import OwlModel


class TrainStepPipeline(StateMachine):
    """Owl level 3

    处理单个 Batch 的完整生命周期（前向、算Loss、反向、更新）。
    """

    # ==========================================
    # 定义状态
    # ==========================================
    started_state = State(StepState.STARTED.value, initial=True)
    grad_zeroed_state = State(StepState.GRAD_ZEROED.value)
    forward_computed_state = State(StepState.FORWARD_COMPUTED.value)
    loss_computed_state = State(StepState.LOSS_COMPUTED.value)
    backward_computed_state = State(StepState.BACKWARD_COMPUTED.value)
    optimized_state = State(StepState.OPTIMIZED.value)
    scheduled_state = State(StepState.SCHEDULED.value)
    ended_state = State(StepState.ENDED.value)

    # ========================================================================
    # 状态转移图
    #
    # +---------------+   +-------------------+   +------------------------+   +---------------------+
    # | started_state |-->| grad_zeroed_state |-->| forward_computed_state |-->| loss_computed_state |
    # +---------------+   +-------------------+   +------------------------+   +---------------------+
    #         ^                                                                           |
    #         |         +-------------+   +-----------------+   +-------------------------+
    #         \---------| ended_state |<--| scheduled_state |<--| backward_computed_state |
    #                   +-------------+   +-----------------+   +-------------------------+
    #
    # ========================================================================
    event_zero_grad = started_state.to(grad_zeroed_state)
    event_forward = grad_zeroed_state.to(forward_computed_state)
    event_compute_loss = forward_computed_state.to(loss_computed_state)
    event_backward = loss_computed_state.to(backward_computed_state)
    event_optimize = backward_computed_state.to(optimized_state)
    event_schedule = optimized_state.to(scheduled_state)
    event_finish = scheduled_state.to(ended_state)

    # 重置
    event_reset_pipeline = ended_state.to(started_state)

    def __init__(
        self,
        model: OwlModel,
        criterion: OwlCriterion,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None=None,
        device:torch.device=torch.device("cpu"),
        non_blocking: bool = True
    ):
        self.nn_model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device

        # 上下文变量 (Context Variables)
        self.ctx_batch: DataSetBatch | None = None
        self.ctx_outputs: Any = None
        self.ctx_model_extra: dict[str, Any] = {}
        self.ctx_loss: torch.Tensor | None = None
        self.ctx_loss_extra: dict[str, Any] = {}
        self.non_blocking = non_blocking
        self.ctx_epoch: int = 0
        self.ctx_step: int = 0

        super().__init__()

    # ==========================================
    # action hook
    # ==========================================

    def on_event_zero_grad(self):
        """清空梯度"""
        self.optimizer.zero_grad()

    def on_event_forward(self):
        """前向传播"""
        self.ctx_outputs = self.nn_model(
            batch_data=self.ctx_batch,
            current_epoch=self.ctx_epoch,
            current_step=self.ctx_step
        )
        self.ctx_model_extra = self.ctx_outputs.get("extra", {})

    def on_event_compute_loss(self):
        """计算损失"""
        loss = self.criterion(
            model_outputs=self.ctx_outputs,
            batch_data=self.ctx_batch,
            current_epoch=self.ctx_epoch,
            current_step=self.ctx_step
        )
        if isinstance(loss, torch.Tensor):
            self.ctx_loss = loss
            self.ctx_loss_extra = {}
        else:
            self.ctx_loss = loss["loss"]
            self.ctx_loss_extra = loss.get("extra", {})

    def on_event_backward(self):
        """反向传播"""
        if self.ctx_loss is None:
            raise RuntimeError("ctx_loss is None before backward.")
        self.ctx_loss.backward()

    def on_event_optimize(self):
        """更新梯度"""
        self.optimizer.step()

    def on_event_schedule(self):
        """学习率调整"""
        if self.scheduler is not None:
            self.scheduler.step()

    def on_event_finish(self):
        """"""
        pass

    def do_step_flow(
        self,
        batch_data: DataSetBatch,
        current_epoch: int = 0,
        current_step: int = 0,
    ) -> dict[str, Any]:
        """

        Args:
            batch_data (DataSetBatch): 原始批次数据。
            current_epoch (int): 当前所属轮次。
            current_step (int): 当前所属全局batch数。
        """
        if self.ended_state.is_active:
            self.event_reset_pipeline()

        self.ctx_epoch = current_epoch
        self.ctx_step = current_step

        batch_data['tp_tensor'] = batch_data['tp_tensor'].to(self.device, non_blocking=self.non_blocking)
        batch_data['gt_tensor'] = batch_data['gt_tensor'].to(self.device, non_blocking=self.non_blocking)
        self.ctx_batch = batch_data

        self.event_zero_grad()
        self.event_forward()
        self.event_compute_loss()
        self.event_backward()
        self.event_optimize()
        self.event_schedule()
        self.event_finish()

        return {
            "loss": self.ctx_loss.item() if self.ctx_loss is not None else 0.0,
            "lr": self.optimizer.param_groups[0]['lr'],
            "metrics":{
                "loss": self.ctx_loss_extra.get("metrics", {}),
                "model": self.ctx_model_extra.get("metrics", {}),
            }
        }