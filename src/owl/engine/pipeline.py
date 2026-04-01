from statemachine import StateMachine, State
from typing import Any, Dict, Optional
import torch

from .state import StepState
from ..toolkits.criterion.base import OwlCriterion
from ..toolkits.data.types import DataSetBatch
from ..toolkits.model.base import OwlModel


class StepPipeline(StateMachine):
    """处理单个 Batch 的完整生命周期（前向、算Loss、反向、更新）。
    """

    # ==========================================
    # 定义状态
    # ==========================================
    started = State(StepState.STARTED.value, initial=True)
    grad_zeroed = State(StepState.GRAD_ZEROED.value)
    forward_computed = State(StepState.FORWARD_COMPUTED.value)
    loss_computed = State(StepState.LOSS_COMPUTED.value)
    backward_computed = State(StepState.BACKWARD_COMPUTED.value)
    optimized = State(StepState.OPTIMIZED.value)
    scheduled = State(StepState.SCHEDULED.value)
    ended = State(StepState.ENDED.value)

    # ==========================================
    # 定义状态转移
    # ==========================================
    run_zero_grad = started.to(grad_zeroed)
    run_forward = grad_zeroed.to(forward_computed)
    run_compute_loss = forward_computed.to(loss_computed)
    run_backward = loss_computed.to(backward_computed)
    run_optimize = backward_computed.to(optimized)
    run_schedule = optimized.to(scheduled)
    run_finish = scheduled.to(ended)

    # 重置
    reset_pipeline = ended.to(started)

    def __init__(self, model: OwlModel,
                 criterion: OwlCriterion,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler.LRScheduler | None=None,
                 device:torch.device=torch.device("cpu"),
                 non_blocking: bool = True):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device

        # 上下文变量 (Context Variables)
        self.ctx_batch: DataSetBatch | None = None
        self.ctx_outputs: Any = None
        self.ctx_loss: torch.Tensor | None = None
        self.non_blocking = non_blocking
        self.ctx_epoch: int = 0
        self.ctx_step: int = 0

        super().__init__()

    # ==========================================
    # action hook
    # ==========================================

    def on_run_zero_grad(self):
        """清空梯度"""
        self.optimizer.zero_grad()

    def on_run_forward(self):
        """前向传播"""
        self.ctx_outputs = self.model(
            batch_data=self.ctx_batch,
            current_epoch=self.ctx_epoch,
            current_step=self.ctx_step
        )

    def on_run_compute_loss(self):
        """计算损失"""
        self.ctx_loss = self.criterion(
            model_outputs=self.ctx_outputs,
            batch_data=self.ctx_batch,
            current_epoch=self.ctx_epoch,
            current_step=self.ctx_step
        )

    def on_run_backward(self):
        """反向传播"""
        self.ctx_loss.backward()

    def on_run_optimize(self):
        """更新梯度"""
        self.optimizer.step()

    def on_run_schedule(self):
        """学习率调整"""
        if self.scheduler is not None:
            self.scheduler.step()

    def on_run_finish(self):
        """"""
        pass

    def do_step_flow(self, batch_data: DataSetBatch, current_epoch: int = 0, current_step: int = 0):
        """

        Args:
            batch_data (DataSetBatch): 原始批次数据。
            current_epoch (int): 当前所属轮次。
            current_step (int): 当前所属全局batch数。
        """
        if self.ended.is_active:
            self.reset_pipeline()

        self.ctx_epoch = current_epoch
        self.ctx_step = current_step

        batch_data['tp_tensors'] = batch_data['tp_tensors'].to(self.device, non_blocking=self.non_blocking)
        batch_data['gt_tensors'] = batch_data['gt_tensors'].to(self.device, non_blocking=self.non_blocking)
        self.ctx_batch = batch_data

        self.run_zero_grad()
        self.run_forward()
        self.run_compute_loss()
        self.run_backward()
        self.run_optimize()
        self.run_schedule()
        self.run_finish()