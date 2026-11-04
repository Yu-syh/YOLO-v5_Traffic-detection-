import torch
import yaml
from pathlib import Path

# 改成你的实际路径
weights_path = r"E:\lanzhi\YOLO\yolov5-master\runs\train\exp17\weights\best.pt"
yaml_path = r"data/traffic/data.yaml"
out_path = str(Path(weights_path).with_name("best_fixed.pt"))

# 1. 用 UTF-8 读取 data.yaml 中正确的 names
with open(yaml_path, encoding="utf-8") as f:
    data = yaml.safe_load(f)

names = data["names"]
if isinstance(names, list):
    names = {i: n for i, n in enumerate(names)}

print("yaml names:", names)

# 2. 加载 checkpoint（PyTorch 2.4 需要 weights_only=False）
ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)

print("ckpt names before:", ckpt["model"].names)

# 3. 替换 model 和 ema 的 names
ckpt["model"].names = names
if ckpt.get("ema") is not None:
    ckpt["ema"].names = names

# 4. 保存修复后的权重
torch.save(ckpt, out_path)
print("saved:", out_path)