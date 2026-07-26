---
name: owl
description: >
  将现有 PyTorch 项目接入 Owl（owl-imdl），并基于当前 Owl 源码构建、调整或排查
  训练与推理工作流。适用于适配 Model、Criterion 和 Dataset，配置 Invocation、
  Optimizer、Scheduler、Checkpoint 与 Workspace，以及诊断调用协议、输出协议和
  运行过程中的问题。
---

# Owl

## 工作原则

将当前项目或当前 Python 环境实际使用的 Owl 源码视为最高事实来源。

Owl 仍可能持续演进。不要根据本 Skill、Asset、记忆或其他版本的示例推测当前 API。
涉及调用签名、Declaration、输出格式、默认值或运行行为时，必须先阅读当前源码。

## 工作流程

1. 理解用户任务和现有项目，确定需要接入、修改或排查的具体边界。尽可能保留用户
   原有的项目结构、组件实现和初始化逻辑。

2. 确定本次任务的 Owl 事实来源：

   - 修改 Owl 本身时，读取当前工作区中的 Owl 源码和测试。
   - 在其他项目中使用 Owl 时，定位当前 Python 环境实际导入的 Owl 包。
   - 如果工作区源码与实际导入版本不同，先判断用户准备修改或运行的是哪一个版本，
     不要混用两者的 API。

   可以使用 Python 定位当前导入的 Owl：

   ```bash
   python -c "from pathlib import Path; import owl; print(Path(owl.__file__).resolve().parent)"
   ```

3. 只围绕当前任务阅读源码。通常按照以下顺序建立理解：

   1. 查看公开导出和 Invocation；
   2. 查看相关类型、协议和组件声明；
   3. 查看 Parser、Validator、Resolver 等边界逻辑；
   4. 查看 Session 或 Runtime 中的实际消费位置；
   5. 查看相关测试和项目示例。

   不要因为 Skill 中出现过某个名称，就假设当前版本仍然存在该类型或目录。

4. 在修改代码前，先给出接入或修复方案，说明：

   - 需要适配哪些组件；
   - 哪些现有实现可以保留；
   - Owl 与用户项目之间的转换发生在哪里；
   - 需要依据源码确认哪些行为；
   - 准备如何进行最小验证。

5. 优先在集成边界使用轻量 Adapter。保留原有 PyTorch Model、Criterion 和权重
   初始化逻辑，只转换 Owl 调用协议、Batch 结构和输出协议所需的部分。

6. 使用当前版本公开的 Invocation API 构建工作流，并通过 `owl.invoke()` 执行。
   除非用户正在开发 Owl 本身，或者排查内部执行问题，否则不要让用户代码直接依赖
   Session 或 Runtime。

7. 逐步验证修改结果：

   - 验证导入和组件构造；
   - 检查一个 Dataset Item；
   - 检查一个整理后的 Batch；
   - 执行一次模型前向调用；
   - 检查模型输出的键、值和形状；
   - 训练任务还要执行 Criterion，并确认最终 Loss 可以反向传播；
   - 最后执行当前条件下最小的 Invocation 冒烟测试。

8. 排查问题时，沿真实执行链路寻找第一个被破坏的边界。找到原因后，只修改恢复
   预期协议所必需的部分，不要顺带重构无关代码。

## 核心边界

### Model 与 Criterion

不要混淆两个 `loss` 的职责：

```text
ModelOutput["loss"]
→ 作为 Criterion 的输入

CriterionOutput["loss"]
→ 作为训练过程中反向传播的最终标量 Loss
```

具体键名、值类型、可选输出和运行时校验规则仍然必须以当前源码为准。

### 权重与 Checkpoint

用户项目原有的预训练权重或外部权重，由原组件或 Adapter 的初始化逻辑负责加载。

不要默认使用 Owl Checkpoint 机制加载任意外部权重。只有 Checkpoint 由 Owl 针对
兼容的组件结构产生时，才按照当前 Owl 源码提供的方式加载模型状态或恢复训练状态。

### 公开 API 与内部实现

构建用户工作流时使用公开 Invocation API 和 `owl.invoke()`。

Session、Runtime、Parser、Resolver 和 Workspace Writer 等内部实现可以用于确认
实际行为和排查问题，但不要在没有必要时将它们变成用户项目的直接依赖。

## Asset

`assets/` 中的文件是可选的起始模板，不是 Owl API 文档，也不是当前版本行为的
事实来源：

- `assets/model.py`：Model 或 Model Adapter 的起始模板；
- `assets/criterion.py`：Criterion 或 Criterion Adapter 的起始模板；
- `assets/train.py`：训练 Invocation 的起始模板；
- `assets/infer.py`：独立推理 Invocation 的起始模板。

只有在与当前任务相关时才使用对应 Asset。使用前必须根据当前 Owl 源码核对其中的
导入路径、调用签名、Declaration 和配置字段。

如果 Asset 与当前源码冲突，以当前源码为准。修改生成结果以适配当前版本，并向用户
说明发现的差异。

如果用户已有对应实现，优先修改或包装现有实现，不要为了使用 Asset 而替换用户代码。

## 约束

- 不要根据旧版本文档、Skill 或 Asset 猜测当前 API。
- 不要维护 Owl 当前类型、字段、默认值或目录结构的重复说明。
- 不要扫描整个 Python 环境，只读取当前任务涉及的 Owl 包和项目文件。
- 不要在未检查实际消费位置时，仅凭 TypeAlias 或 Protocol 推断运行行为。
- 不要用 Owl Checkpoint 机制替代外部模型原有的权重加载方式。
- 不要为了接入 Owl 重写本来可以通过 Adapter 保留的组件。
- 不要让普通用户代码无必要地依赖 Session、Runtime 或其他内部实现。
- 只进行完成用户任务所需的最小修改。