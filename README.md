# 道路病害检测系统

基于 **YOLOv5** 的道路病害检测项目。训练完成的模型封装在两个界面中：

- **Flask 后端 + 浏览器网页**（[`app.py`](./app.py)）—— 零安装，任何人访问网址即可使用
- **PyQt5 桌面前端**（[`gui.py`](./gui.py)）—— 支持图片、视频、摄像头实时检测，会自动拉起本机后端

本仓库只包含代码，**不包含数据集和训练好的权重**（文件体积超出 GitHub 限制）。下面说明如何获取权重并运行。

---

## 功能概览

- 识别 19 类道路病害与路侧目标（中文类别名）：

| 类别 | — | — |
| --- | --- | --- |
| 坑槽 | 块状修补 | 横向修补 |
| 纵向修补 | 网状修补 | 纵向裂纹 |
| 横向裂纹 | 纵向裂缝 | 网状裂纹 |
| 松散 | 抛洒物 | 标线模糊 |
| 护栏损坏 | 方形雨水盖 | 圆形雨水盖 |
| 圆形井盖 | 圆形桶 | 锥形桶 |
| 人 | — | — |

- 标注图使用中文标签（Windows 自动使用 `simhei.ttf`；Linux / macOS 会回退到系统默认字体）
- 推理串行加锁，Flask 多线程与 PyQt 多请求都能安全运行
- 自动记录预处理 / 推理 / NMS 分段耗时

---

## 目录结构

```
yolov5-master/
├── app.py                    # Flask 后端 + 网页界面（本项目新增）
├── gui.py                    # PyQt5 桌面前端（本项目新增）
├── train.py / val.py / detect.py
├── models/                   # YOLOv5 模型定义
├── utils/                    # YOLOv5 工具函数
├── runs/train/exp17/weights/ # 训练权重（需自行放置，见下文）
└── requirements.txt          # 已追加 flask 与 PyQt5
```

项目是在 Ultralytics YOLOv5 v7.0 之上扩展的，原始 YOLOv5 的训练、验证、导出命令（`train.py` / `val.py` / `export.py`）仍然完整可用。

---

## 环境要求

| 项目 | 版本 |
| --- | --- |
| Python | >= 3.8 |
| PyTorch | >= 1.8.0 |
| CUDA（可选） | 建议 11.8 / 12.x，推理更快 |
| 操作系统 | Windows 首选；macOS / Linux 仅需额外处理中文字体 |

GPU 安装：到 <https://pytorch.org/get-started/locally/> 选择对应命令。CPU 安装：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

其他依赖（含 flask、PyQt5）：

```bash
pip install -r requirements.txt
```

仅使用网页后端、不需要桌面界面时，可以跳过 PyQt5：

```bash
pip install flask
```

---

## 获取权重

仓库 `.gitignore` 默认忽略 `*.pt`，权重未上传。请按以下任一方式获取 `best_fixed.pt`：

1. **自行训练（推荐）**：准备自己的数据集（YOLO 格式），放入 `data/` 目录后执行：

   ```bash
   python train.py --imgsz 640 --batch 32 --epochs 200 --weights yolov5n.pt --cfg yolov5n.yaml
   ```

   训练完成后权重保存在 `runs/train/expX/weights/best.pt`，把它改名为 `best_fixed.pt` 并放到默认路径（或用环境变量指定）。

2. **从作者处获取**：如果你是从课程 / 演示 / 联系作者获得预训练权重，下载后放入：

   ```
   yolov5-master/runs/train/exp17/weights/best_fixed.pt
   ```

   或者通过环境变量指定任意位置：

   ```bash
   set ROAD_WEIGHTS=D:\path\to\best_fixed.pt          # Windows PowerShell
   # export ROAD_WEIGHTS=/home/you/best_fixed.pt       # macOS / Linux
   ```

3. **只跑通流程**：如果暂时没有自定义权重，也可以先用官方 YOLOv5 权重快速验证后端能否启动：

   ```bash
   python app.py --weights yolov5n.pt --device cpu
   ```

   此时检测类别为 COCO 80 类，不含道路病害。

---

## 快速开始

### 方式一：Flask 网页（推荐公网使用）

本地：

```bash
python app.py --host 0.0.0.0 --port 5000
```

浏览器访问 <http://127.0.0.1:5000> 即可上传图片检测。

参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--weights` | `runs/train/exp17/weights/best_fixed.pt` | 权重路径 |
| `--device` | 自动选择 GPU / CPU | `0` 表示 `cuda:0`，`cpu` 表示强制 CPU |
| `--imgsz` | `640` | 推理输入尺寸 |
| `--host` | `127.0.0.1` | 绑定地址，公网穿透用 `0.0.0.0` |
| `--port` | `5000` | 监听端口 |

等价环境变量：`ROAD_WEIGHTS` / `ROAD_DEVICE` / `ROAD_IMGSZ` / `ROAD_HOST` / `ROAD_PORT`。

### 方式二：PyQt5 桌面前端（推荐本地演示）

```bash
python gui.py
```

前端会自动探测本机 `http://127.0.0.1:5000`，若后端未运行则用**当前 Python 解释器**自动拉起 `app.py`。状态栏显示「后端：就绪」后即可：

- **选择图片** —— 单次检测 JPG / PNG / BMP
- **选择视频** —— MP4 / AVI / MOV / MKV，逐帧检测
- **开启摄像头** —— 默认索引 0，Windows 自动使用 `CAP_DSHOW`
- **停止检测 / 重试连接后端** —— 中断当前任务或重新拉起后端

也可以让 GUI 连接远程后端：设置环境变量 `ROAD_API_URL=https://你的公网地址`。

### 方式三：REST API 调用

```bash
curl -X POST http://127.0.0.1:5000/api/detect \
  -F "image=@test.jpg" \
  -F "conf=0.25" \
  -F "iou=0.45"
```

响应字段：

| 字段 | 说明 |
| --- | --- |
| `success` | 是否成功 |
| `total` | 检测目标总数 |
| `detections[].class_name` / `confidence` / `bbox` | 类别、置信度、x1y1x2y2 |
| `counts` | 按类别聚合的数量（字典） |
| `elapsed_ms.{preprocess,inference,nms,total}` | 分段耗时（毫秒） |
| `annotated` | 标注图 `data:image/jpeg;base64,...`，可直接作为 `<img src>` |

健康检查：

```bash
curl http://127.0.0.1:5000/api/health
```

- 首次加载模型期间返回 `{"status":"loading"}`（HTTP 503）
- 就绪后返回 `{"status":"ok","model":{...}}`
- 权重缺失返回 `{"status":"error","error":"权重不存在: ..."}`（HTTP 503）

---

## 公网部署（让别人访问网址即可使用）

### 临时演示（免费穿透，数分钟启动）

启动后端监听所有网卡：

```bash
python app.py --host 0.0.0.0 --port 5000
```

再开一个终端执行穿透，任选其一：

```bash
# cloudflared（推荐，无需注册）
cloudflared tunnel --url http://127.0.0.1:5000

# cpolar（国内访问较快）
cpolar http 5000

# ngrok（需先配置 token）
ngrok http 5000
```

工具会输出一个 `https://xxx.trycloudflare.com` 之类的临时地址，把它发给别人即可。

> cloudflared Windows 用户：winget 安装后若 PATH 未刷新，直接用完整路径：
> `"C:\Program Files (x86)\cloudflared\cloudflared.exe" tunnel --url http://127.0.0.1:5000`

### 正式部署（云服务器）

后端兼容 WSGI，在 Linux 上建议用 `waitress`（Windows 也可用）或 `gunicorn`：

```bash
# Windows
pip install waitress
waitress-serve --host 0.0.0.0 --port 5000 app:app

# Linux
pip install gunicorn
gunicorn -w 1 -b 0.0.0.0:5000 app:app
```

> 注意：多 worker 时每个 worker 都会各自加载一份模型，GPU 上会 OOM；单 worker 足以承载串行推理需求。

再用 nginx 反向代理 + HTTPS（Let's Encrypt / Cloudflare），绑定自己的域名即可长期提供服务。

---

## 迁移 / 部署时需要哪些文件

| 文件 | 必须带 | 说明 |
| --- | --- | --- |
| `app.py` | ✅ | Flask 后端 + 网页界面 |
| `gui.py` | 可选 | 仅桌面端需要 |
| `requirements.txt` | 可选 | 目标项目已有依赖清单就不覆盖 |
| `runs/train/exp17/weights/best_fixed.pt` | ✅ | 权重，或用 `ROAD_WEIGHTS` 指定 |
| `models/`、`utils/` | ✅ | YOLOv5 核心代码，`app.py` 会导入 |

如果目标环境已经是一份 YOLOv5 仓库，**只需复制 `app.py` + `gui.py` + 权重** 即可。

---

## 常见问题

**权重报错 `FileNotFoundError`**
- 默认找 `runs/train/exp17/weights/best_fixed.pt`；改路径请用 `ROAD_WEIGHTS` 环境变量或 `--weights` 参数。

**公网打开显示 502 / 无法连接**
- 后端是否仍在运行？PyCharm Run 窗口和穿透终端都不能关。
- 后端是否在 `0.0.0.0` 上监听？`--host 0.0.0.0` 或 `ROAD_HOST=0.0.0.0`。
- 穿透工具是否返回了 `https://...` 地址？等出现隧道链接再测试。

**中文标签变方块 / 英文**
- Windows 默认使用 `C:/Windows/Fonts/simhei.ttf`。
- Linux：安装 `fonts-wqy-zenhei` 或把 `simhei.ttf` 放到项目根目录；`app.py` 中 `FONT` 变量会优先使用项目内字体文件。
- macOS：建议把黑体（如 `STHeiti Light.ttc`）放到项目根目录，或修改 `FONT = "/System/Library/Fonts/PingFang.ttc"`。

**PyQt 摄像头打不开**
- Windows 默认使用 `CAP_DSHOW`，确保摄像头未被其他程序占用。
- 可手动改 `gui.py` 中 `cv2.VideoCapture(0, cv2.CAP_DSHOW)` 为 `cv2.VideoCapture(0)` 再试。

**GPU 推理报错 / 速度慢**
- 检查 `torch.cuda.is_available()`；没有 GPU 时加 `ROAD_DEVICE=cpu` 强制走 CPU。
- `CUDA out of memory`：减小 `ROAD_IMGSZ` 或使用更小的权重（如 `yolov5n`）。

**公网多人使用卡顿**
- 模型是串行推理，多人会排队；要加并发请用 Redis 队列或多进程 + 多 GPU。
- 本项目仅用于演示 / 课程 / 小规模场景。

---

## 许可证

- YOLOv5 原始仓库：AGPL-3.0，见 [LICENSE](./LICENSE)。
- 本项目新增的 `app.py`、`gui.py` 遵循相同许可条款。
- 训练所用数据集与预训练权重，请遵循其各自的使用协议。
