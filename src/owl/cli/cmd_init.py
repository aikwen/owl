from pathlib import Path
import shutil
import questionary
from questionary import Choice

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

def func(args):
    choices = [
        Choice(title="训练 (train.yaml)", value="train.yaml"),
        Choice(title="验证 (validate_metric.yaml)", value="validate_metric.yaml"),
        Choice(title="鲁棒性 (validate_robust.yaml)", value="validate_robust.yaml"),
        Choice(title="可视化 (visualization.yaml)", value="visualization.yaml"),
    ]
    selected_template = questionary.select(
        message="请选择一个要生成的模板 (上下键移动，回车确认):",
        choices=choices
    ).ask()

    if not selected_template:
        return

    # 当前工作目录
    current_cwd = Path.cwd()
    # 选择的模板文件路径
    src_path = TEMPLATE_DIR / selected_template
    # 生成的模板文件
    target_filename = selected_template
    dst_path = current_cwd / target_filename

    # 检查模板源文件是否存在
    if not src_path.exists():
        print(f"内部模板丢失: {src_path}")
        return

    # 文件名重复检测循环
    while dst_path.exists():
        print(f"文件 '{target_filename}' 在当前目录下已存在。")

        # 让用户输入新名字
        new_name = questionary.text(
            "请输入新的文件名 (留空则取消):",
            default=f"new_{target_filename}"  # 给个默认建议，体验更好
        ).ask()

        # 如果用户没输入直接取消
        if not new_name or not new_name.strip():
            print("操作已取消。")
            return

        # 更新目标文件名和路径，准备下一次循环检测
        target_filename = new_name.strip()
        # 补全后缀 (如果用户忘了写 .yaml)
        if not target_filename.endswith(".yaml"):
            target_filename += ".yaml"

        dst_path = current_cwd / target_filename

    # 执行复制
    try:
        shutil.copy(src_path, dst_path)
        print(f"\n[成功] 已生成文件: {dst_path.name}")
        print(f"   路径: {dst_path}")
    except Exception as e:
        print(f"[失败] 无法写入文件: {e}")