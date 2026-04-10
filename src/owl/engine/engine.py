from typing import Any
import torch
from torch.utils.data import DataLoader

from statemachine import StateMachine, State
from .state import ExecMode, ExecState, StepState
from .pipeline import StepPipeline
from ..toolkits.model.base import OwlModel
from ..toolkits.criterion.base import OwlCriterion
from ..toolkits.visual.base import OwlVisualizer
from ..toolkits.data.types import DataSetBatch

class OwlEngine(StateMachine):
    """Owl level 2

    """

    # ==========================================
    # 状态定义
    # ==========================================
    start_state = State(ExecState.START.value, initial=True)
    routing_state = State(ExecState.ROUTING.value)
    train_state = State(ExecState.TRAIN.value)
    validate_state = State(ExecState.VALIDATE.value)
    visual_state = State(ExecState.VISUAL.value)
    end_state = State(ExecState.END.value, final=True)

    # ========================================================================
    # 状态转移图
    #
    #                                      +----------------+
    #                                  /-->| validate_state |----\
    #                                 /    +----------------+     \
    #                                /       |            ^        \
    #                               /        v            |         v
    # +-------------+   +---------------+  +----------------+    +-----------+
    # | start_state |-->| routing_state |->|  train_state   |    | end_state |
    # +-------------+   +---------------+  +----------------+    +-----------+
    #                               \                            ^
    #                                \     +----------------+   /
    #                                 \--->|  visual_state  |--/
    #                                      +----------------+
    # ========================================================================

    event_start_to_routing = start_state.to(routing_state)

    # 路由
    event_route_to_train = routing_state.to(train_state)
    event_route_to_validate = routing_state.to(validate_state)
    event_route_to_visual = routing_state.to(visual_state)

    # Train 与 Validate 的双向流转
    event_train_to_validate = train_state.to(validate_state)
    event_validate_to_train = validate_state.to(train_state)

    # 指向结束
    event_train_to_end = train_state.to(end_state)
    event_validate_to_end = validate_state.to(end_state)
    event_visual_to_end = visual_state.to(end_state)

    def __init__(self,
                 model: OwlModel,
                 criterion: OwlCriterion | None,
                 optimizer: torch.optim.Optimizer | None,
                 scheduler: Any | None,
                 train_loader: DataLoader | None,
                 val_loaders:  dict[str, DataLoader] | None,
                 visualizer:   OwlVisualizer | None = None,):

        # 必要的组件
        self.model: OwlModel = model
        self.criterion:  OwlCriterion | None = criterion
        self.optimizer:  torch.optim.Optimizer | None = optimizer
        self.scheduler:  Any | None = scheduler
        self.visualizer: OwlVisualizer | None  = visualizer

        # 数据集
        self.train_loader: DataLoader | None = train_loader
        self.val_loaders: dict[str, DataLoader] | None = val_loaders or {}

        # 设备
        self.device: torch.device = torch.device("cpu")

        self.pipeline: StepPipeline | None = None
        # 运行时上下文
        self.current_mode: ExecMode | None = None
        self.current_epoch: int = 0
        self.current_step:  int = 0
        self.max_epochs:    int = 1

        super().__init__()


    def run(self, mode: ExecMode,
            max_epochs:  int = 1,
            start_epoch: int = 0,
            device: torch.device | str = torch.device("cpu")):
        """总入口"""
        self.current_mode = mode
        self.max_epochs = max_epochs
        self.current_epoch = start_epoch
        self.current_step = 0
        self.device = torch.device(device)

        # Start -> routing_state
        self.event_start_to_routing()

        if mode == ExecMode.TRAIN:
            self.event_route_to_train()
        elif mode == ExecMode.VALIDATE:
            self.event_route_to_validate()
        elif mode == ExecMode.VISUALIZE:
            self.event_route_to_visual()

        # 无限循环状态机驱动
        while not self.end_state.is_active:

            # 处于 TRAIN 节点
            if self.train_state.is_active:
                """
                执行一轮训练，直接进入 validate_state
                """
                self._do_train_epoch()
                self.event_train_to_validate()

            # 处于 VALIDATE 节点
            elif self.validate_state.is_active:
                """
                 执行 validate
                 if 当前模式 == TRAIN:
                    epoch++
                    if epoch < self.max_epochs:
                        进入 train_state
                 else:
                    进入 end_state
                """
                self._do_validate()

                if self.current_mode == ExecMode.TRAIN:
                    self.current_epoch += 1
                    # 如果当前是 train 模式, 同时 epoch 没有结束
                    if self.current_epoch < self.max_epochs:
                        self.event_validate_to_train()
                    else:
                        self.event_validate_to_end()
                else:
                    self.event_validate_to_end()

            # 处于 VISUAL 节点
            elif self.visual_state.is_active:
                self._do_visualize()
                self.event_visual_to_end()

    def on_event_route_to_train(self):
        """初始化 pipeline
        :return:
        """
        self.pipeline = StepPipeline(
            model=self.model,
            criterion=self.criterion,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device,
            non_blocking=True,
        )

    def _do_train_epoch(self):
        self.model.train()
        for batch_id, batch_data in enumerate(self.train_loader):
            batch_data: DataSetBatch
            self.current_step += 1
            self.pipeline.do_step_flow(batch_data, self.current_epoch, self.current_step)

    def _do_validate(self):
        self.model.eval()
        with torch.no_grad():
            for dataset_name, dataloader in self.val_loaders.items():
                for batch_data in dataloader:
                    batch_data: DataSetBatch
                    batch_data['tp_tensors'] = batch_data['tp_tensors'].to(self.device, non_blocking=True,)
                    batch_data['gt_tensors'] = batch_data['gt_tensors'].to(self.device, non_blocking=True,)
                    # 获取输出
                    outputs = self.model(
                        batch_data,
                        current_epoch=self.current_epoch,
                        current_step=self.current_step
                    )
                    # TODO:计算 AUC，F1-score .... 之类的


    def _do_visualize(self):
        self.model.eval()
        with torch.no_grad():
            for _, dataloader in self.val_loaders.items():
                for batch_data in dataloader:
                    batch_data: DataSetBatch
                    batch_data['tp_tensors'] = batch_data['tp_tensors'].to(self.device, non_blocking=True, )
                    batch_data['gt_tensors'] = batch_data['gt_tensors'].to(self.device, non_blocking=True, )
                    # 模型输出
                    outputs = self.model(
                        batch_data,
                        current_epoch=self.current_epoch,
                        current_step=self.current_step
                    )
                    if self.visualizer:
                        _pred_masks = self.visualizer(outputs)
                        # TODO:保存图片到文件夹