import torch
from torchvision import models
import re

def add_hidden_relu_to_repr(model_repr: str) -> str:
    lines = model_repr.splitlines()
    new_lines = []
    for i, line in enumerate(lines):
        new_lines.append(line)

        # 获取缩进层级
        indent_match = re.match(r"^(\s*)", line)
        indent = indent_match.group(1) if indent_match else ""

        # ---- 仅在 Bottleneck 内部添加 ----
        # 6个空格或以上缩进通常是 Bottleneck 内部
        in_bottleneck = len(indent) >= 6

        # bn1/bn2 都要加
        if in_bottleneck and (("(bn1):" in line) or ("(bn2):" in line)):
            # 下一个行不是 ReLU 才加，避免重复
            if i + 1 >= len(lines) or "ReLU" not in lines[i + 1]:
                new_lines.append(f"{indent}(relu): ReLU(inplace=True)")

    return "\n".join(new_lines)


def print_model_info_with_hidden_relu(model, filename="model_info_ResNet152.txt"):
    """
    输出包含隐式 ReLU 的模型信息
    """
    original_repr = str(model)
    enhanced_repr = add_hidden_relu_to_repr(original_repr)

    with open(filename, "w") as f:
        # print("Model architecture with added implicit ReLU:\n", file=f)
        print(enhanced_repr, file=f)

    print(f"✅ Saved enhanced model info to {filename}")


if __name__ == "__main__":
    model = models.resnet152(pretrained=False)
    print_model_info_with_hidden_relu(model)
