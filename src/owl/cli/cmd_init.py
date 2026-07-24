from pathlib import Path
import shutil
import questionary
from questionary import Choice

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

def func(args):
    choices = [
        Choice(title="训练", value="train.py"),
        Choice(title="微调", value="finetune.py"),
        Choice(title="验证", value="val_metric.py"),
        Choice(title="可视化", value="visual.py"),
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
            default=f"{target_filename}"
        ).ask()

        # 如果用户没输入直接取消
        if not new_name or not new_name.strip():
            print("操作已取消。")
            return

        # 更新目标文件名和路径并补全后缀
        target_filename = new_name.strip()
        if not target_filename.endswith(".py"):
            target_filename += ".py"

        dst_path = current_cwd / target_filename

    # 复制
    try:
        shutil.copy(src_path, dst_path)
        print(f"\n[success] 已生成文件: {dst_path.name}")
        print(f"   路径: {dst_path}")
    except Exception as e:
        print(f"[fail] 无法写入文件: {e}")