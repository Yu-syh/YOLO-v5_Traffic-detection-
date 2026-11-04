# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""道路病害检测 Flask 后端与网页界面。

启动时加载一次 YOLOv5 权重，既提供浏览器检测页面，也提供 API 供 PyQt 前端调用。

启动:
    python app.py --weights runs/train/exp17/weights/best_fixed.pt --device 0

接口:
    GET  /             浏览器检测页面
    GET  /api/health   后端、模型与类别信息
    POST /api/detect   multipart 上传 image，可选 conf / iou，返回检测框与标注图

WSGI 部署（如 waitress / gunicorn）导入 app:app 时会按需加载模型，首次访问稍慢。
"""

import argparse
import base64
import io
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request
from PIL import Image, ImageOps
from ultralytics.utils.plotting import Annotator, colors

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.torch_utils import select_device, smart_inference_mode

FILE = Path(__file__).resolve()
ROOT = FILE.parent
WEIGHTS = ROOT / "runs/train/exp17/weights/best_fixed.pt"
FONT = "C:/Windows/Fonts/simhei.ttf" if Path("C:/Windows/Fonts/simhei.ttf").exists() else "Arial.ttf"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64 MB
app.json.ensure_ascii = False

_detector = None
_detector_lock = threading.Lock()
_init_error = None
_preload_thread = None

MODEL_WEIGHTS = os.getenv("ROAD_WEIGHTS") or str(WEIGHTS if WEIGHTS.exists() else WEIGHTS.with_name("best.pt"))
MODEL_DEVICE = os.getenv("ROAD_DEVICE", "")
MODEL_IMGSZ = int(os.getenv("ROAD_IMGSZ", 640))


class RoadDetector:
    """封装 YOLOv5 道路病害模型，串行执行推理以保证 Flask 多线程下安全。"""

    def __init__(self, weights, device="", imgsz=640):
        self.weights = str(weights)
        self.device = select_device(device)
        self.model = DetectMultiBackend(self.weights, device=self.device)
        self.names = self.model.names
        self.stride = self.model.stride
        self.imgsz = check_img_size(imgsz, s=self.stride)
        self.lock = threading.Lock()
        self.model.warmup(imgsz=(1, 3, self.imgsz, self.imgsz))

    @smart_inference_mode()
    def detect(self, im0, conf_thres=0.25, iou_thres=0.45, max_det=1000):
        """对 BGR 图像执行检测，返回 (检测结果列表, 标注图, 分段耗时毫秒)。"""
        with self.lock:
            t0 = time.perf_counter()
            im = letterbox(im0, self.imgsz, stride=self.stride, auto=True)[0]
            im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR -> RGB, HWC -> CHW
            im = np.ascontiguousarray(im)

            t1 = time.perf_counter()
            im = torch.from_numpy(im).to(self.device).float() / 255
            im = im.unsqueeze(0)
            pred = self.model(im)
            if isinstance(pred, (list, tuple)):  # 部分导出格式返回 (pred, raw)
                pred = pred[0]
            t2 = time.perf_counter()
            det = non_max_suppression(pred, conf_thres, iou_thres, max_det=max_det)[0]
            t3 = time.perf_counter()

            results = []
            annotated = im0
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                annotator = Annotator(im0.copy(), line_width=3, font=FONT, pil=True, example=str(self.names))
                for *xyxy, conf, cls in det:
                    c = int(cls)
                    name = self.names[c]
                    box = [int(v) for v in xyxy]
                    results.append(
                        {
                            "class_id": c,
                            "class_name": name,
                            "confidence": round(float(conf), 4),
                            "bbox": box,
                        }
                    )
                    annotator.box_label(xyxy, f"{name} {float(conf):.2f}", color=colors(c, True))
                annotated = annotator.result()
            t4 = time.perf_counter()

        results.sort(key=lambda r: r["confidence"], reverse=True)
        elapsed = {
            "preprocess_ms": round((t1 - t0) * 1000, 1),
            "inference_ms": round((t2 - t1) * 1000, 1),
            "nms_ms": round((t3 - t2) * 1000, 1),
            "total_ms": round((t4 - t0) * 1000, 1),
        }
        return results, annotated, elapsed

    def info(self):
        return {
            "weights": self.weights,
            "device": str(self.device),
            "imgsz": self.imgsz,
            "names": [self.names[i] for i in sorted(self.names)],
        }


def init_detector(weights=None, device=None, imgsz=None):
    """按需加载模型；重复调用直接返回已加载实例，供脚本和 WSGI 部署共用。"""
    global _detector, _init_error
    if _detector is not None:
        return _detector
    with _detector_lock:
        if _detector is not None:
            return _detector
        weight_path = str(weights or MODEL_WEIGHTS)
        if not Path(weight_path).exists():
            _init_error = f"权重不存在: {weight_path}"
            raise FileNotFoundError(_init_error)
        try:
            _detector = RoadDetector(weight_path, MODEL_DEVICE if device is None else device, imgsz or MODEL_IMGSZ)
        except Exception as e:
            _init_error = str(e)
            raise
        print(
            f"模型就绪: device={_detector.device} imgsz={_detector.imgsz} classes={len(_detector.names)}",
            flush=True,
        )
    return _detector


def _preload():
    try:
        init_detector()
    except Exception as e:
        print(f"模型加载失败: {e}", flush=True)


def _start_preload():
    """WSGI 场景下由首个健康检查触发后台加载，避免阻塞请求。"""
    global _preload_thread
    if _preload_thread is not None and _preload_thread.is_alive():
        return
    if _detector is not None or _init_error:
        return
    _preload_thread = threading.Thread(target=_preload, name="model-preload", daemon=True)
    _preload_thread.start()


HTML_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>道路病害检测</title>
<style>
  :root { --bg:#f3f4f6; --card:#fff; --text:#111827; --muted:#6b7280; --line:#e5e7eb;
          --primary:#2563eb; --primary-dark:#1d4ed8; --error:#b91c1c; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
         font-family:"Segoe UI","Microsoft YaHei",system-ui,sans-serif; }
  .wrap { max-width:1180px; margin:0 auto; padding:28px 18px 56px; }
  header h1 { margin:0 0 6px; font-size:24px; }
  header p { margin:0 0 22px; color:var(--muted); font-size:14px; }
  .grid { display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1.2fr); gap:18px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:12px;
          padding:18px; box-shadow:0 1px 2px rgba(0,0,0,.04); }
  .drop { border:2px dashed #cbd5e1; border-radius:10px; padding:26px 16px;
          text-align:center; cursor:pointer; transition:.15s; }
  .drop:hover, .drop.over { border-color:var(--primary); background:#eff6ff; }
  .drop p { margin:0 0 6px; font-weight:600; }
  .drop small { color:var(--muted); }
  .preview img { width:100%; max-height:280px; object-fit:contain; border-radius:8px;
                 background:#0f172a; display:none; margin-top:14px; }
  .params { margin-top:18px; display:grid; gap:12px; }
  label { display:flex; justify-content:space-between; font-size:13px; color:var(--muted); margin-bottom:6px; }
  input[type=range] { width:100%; accent-color:var(--primary); }
  button { width:100%; margin-top:18px; padding:11px 16px; border:0; border-radius:8px;
           background:var(--primary); color:#fff; font-size:15px; font-weight:600; cursor:pointer; }
  button:hover { background:var(--primary-dark); }
  button:disabled { background:#9ca3af; cursor:not-allowed; }
  .error { min-height:18px; margin:12px 0 0; color:var(--error); font-size:13px; }
  .result-head { display:flex; justify-content:space-between; align-items:baseline; gap:12px; }
  .result-head h2 { margin:0; font-size:18px; }
  .muted { color:var(--muted); font-size:13px; }
  .summary { margin:12px 0; min-height:22px; }
  .chips { display:flex; flex-wrap:wrap; gap:8px; margin-top:8px; }
  .chip { background:#eff6ff; color:#1e40af; border-radius:999px; padding:4px 10px; font-size:12px; }
  #annotated { width:100%; border-radius:10px; margin-top:12px; display:none; background:#0f172a; }
  table { width:100%; border-collapse:collapse; margin-top:14px; font-size:13px; display:none; }
  th, td { text-align:left; padding:8px 10px; border-bottom:1px solid var(--line); }
  th { color:var(--muted); font-weight:600; }
  td:last-child { color:var(--muted); }
  footer { margin-top:28px; color:var(--muted); font-size:12px; text-align:center; }
  @media (max-width:900px) { .grid { grid-template-columns:1fr; } }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>道路病害检测</h1>
    <p>上传道路图片，自动识别坑槽、裂纹、修补等病害。</p>
  </header>
  <main class="grid">
    <section class="card">
      <div class="drop" id="drop">
        <p>点击选择图片，或拖拽到这里</p>
        <small>支持 JPG / PNG / BMP，单张不超过 64 MB</small>
      </div>
      <input type="file" id="file" accept="image/*" hidden>
      <div class="preview"><img id="preview" alt="待检测图片预览"></div>
      <div class="params">
        <div>
          <label for="conf"><span>置信度阈值</span><span id="confValue">0.25</span></label>
          <input type="range" id="conf" min="0.05" max="0.9" step="0.01" value="0.25">
        </div>
        <div>
          <label for="iou"><span>NMS IoU</span><span id="iouValue">0.45</span></label>
          <input type="range" id="iou" min="0.1" max="0.9" step="0.01" value="0.45">
        </div>
      </div>
      <button id="run" disabled>开始检测</button>
      <p class="error" id="error"></p>
    </section>
    <section class="card">
      <div class="result-head"><h2>检测结果</h2><span class="muted" id="latency"></span></div>
      <div class="summary" id="summary">等待检测…</div>
      <img id="annotated" alt="检测结果">
      <table id="table">
        <thead><tr><th>类别</th><th>置信度</th><th>位置 (x1,y1,x2,y2)</th></tr></thead>
        <tbody id="tbody"></tbody>
      </table>
    </section>
  </main>
  <footer>YOLOv5 道路病害检测</footer>
</div>
<script>
const drop = document.getElementById("drop");
const input = document.getElementById("file");
const preview = document.getElementById("preview");
const run = document.getElementById("run");
const error = document.getElementById("error");
const conf = document.getElementById("conf");
const iou = document.getElementById("iou");
let selected = null;
let busy = false;

drop.addEventListener("click", () => input.click());
input.addEventListener("change", (event) => selectFile(event.target.files[0]));
["dragenter", "dragover"].forEach((name) => drop.addEventListener(name, (event) => {
  event.preventDefault();
  drop.classList.add("over");
}));
["dragleave", "drop"].forEach((name) => drop.addEventListener(name, (event) => {
  event.preventDefault();
  drop.classList.remove("over");
}));
drop.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  if (file) selectFile(file);
});
conf.addEventListener("input", () => document.getElementById("confValue").textContent = Number(conf.value).toFixed(2));
iou.addEventListener("input", () => document.getElementById("iouValue").textContent = Number(iou.value).toFixed(2));
run.addEventListener("click", detect);

function selectFile(file) {
  if (!file.type.startsWith("image/")) { error.textContent = "请选择图片文件"; return; }
  if (file.size > 64 * 1024 * 1024) { error.textContent = "图片超过 64 MB"; return; }
  selected = file;
  error.textContent = "";
  preview.src = URL.createObjectURL(file);
  preview.style.display = "block";
  run.disabled = false;
}

async function detect() {
  if (!selected || busy) return;
  busy = true;
  run.disabled = true;
  run.textContent = "检测中…";
  error.textContent = "";
  const body = new FormData();
  body.append("image", selected);
  body.append("conf", conf.value);
  body.append("iou", iou.value);
  try {
    const response = await fetch("/api/detect", { method: "POST", body });
    const data = await response.json();
    if (!response.ok || !data.success) throw new Error(data.error || "HTTP " + response.status);
    render(data);
  } catch (e) {
    error.textContent = "检测失败：" + e.message;
  } finally {
    busy = false;
    run.disabled = false;
    run.textContent = "开始检测";
  }
}

function render(data) {
  const annotated = document.getElementById("annotated");
  if (data.annotated) {
    annotated.src = data.annotated;
    annotated.style.display = "block";
  }
  const table = document.getElementById("table");
  const tbody = document.getElementById("tbody");
  const summary = document.getElementById("summary");
  tbody.innerHTML = "";
  table.style.display = data.detections.length ? "table" : "none";
  if (!data.detections.length) {
    summary.textContent = "未检测到道路病害";
  } else {
    let chips = "";
    for (const [name, count] of Object.entries(data.counts)) {
      chips += '<span class="chip">' + name + " × " + count + "</span>";
    }
    summary.innerHTML = "<strong>共检测到 " + data.total + " 处目标</strong><div class=\\"chips\\">" + chips + "</div>";
    for (const det of data.detections) {
      const row = document.createElement("tr");
      row.innerHTML = "<td>" + det.class_name + "</td><td>" + det.confidence.toFixed(2) +
        "</td><td>" + det.bbox.join(", ") + "</td>";
      tbody.appendChild(row);
    }
  }
  const elapsed = data.elapsed_ms;
  document.getElementById("latency").textContent =
    "耗时 " + elapsed.total_ms.toFixed(1) + " ms（推理 " + elapsed.inference_ms.toFixed(1) + " ms）";
}
</script>
</body>
</html>
"""


@app.get("/")
def index():
    return HTML_PAGE


@app.get("/api/health")
def health():
    if _detector is None:
        if _init_error:
            return jsonify({"status": "error", "error": _init_error}), 503
        _start_preload()
        return jsonify({"status": "loading"}), 503
    return jsonify({"status": "ok", "model": _detector.info()})


@app.post("/api/detect")
def detect():
    file = request.files.get("image")
    if file is None or not file.filename:
        return jsonify({"success": False, "error": "缺少 image 文件"}), 400

    try:
        conf = min(max(float(request.form.get("conf", 0.25)), 0.01), 0.99)
        iou = min(max(float(request.form.get("iou", 0.45)), 0.01), 0.99)
    except ValueError:
        return jsonify({"success": False, "error": "conf/iou 必须是 0-1 的数字"}), 400

    raw = file.read()
    try:
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
        im0 = np.array(image)[:, :, ::-1].copy()  # RGB -> BGR
    except Exception as e:
        return jsonify({"success": False, "error": f"图片解码失败: {e}"}), 415

    try:
        detector = init_detector()
    except Exception as e:
        return jsonify({"success": False, "error": f"模型加载失败: {e}"}), 500

    try:
        detections, annotated, elapsed = detector.detect(im0, conf, iou)
    except Exception as e:
        return jsonify({"success": False, "error": f"推理失败: {e}"}), 500

    ok, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    annotated_uri = "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode() if ok else ""

    counts = {}
    for d in detections:
        counts[d["class_name"]] = counts.get(d["class_name"], 0) + 1

    return jsonify(
        {
            "success": True,
            "total": len(detections),
            "detections": detections,
            "counts": dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True)),
            "image_size": [im0.shape[1], im0.shape[0]],
            "elapsed_ms": elapsed,
            "annotated": annotated_uri,
        }
    )


@app.errorhandler(413)
def too_large(_):
    return jsonify({"success": False, "error": "图片过大（上限 64 MB）"}), 413


def parse_opt():
    default_weights = WEIGHTS if WEIGHTS.exists() else WEIGHTS.with_name("best.pt")
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=os.getenv("ROAD_WEIGHTS", str(default_weights)))
    parser.add_argument("--device", default=os.getenv("ROAD_DEVICE", ""), help="cuda device 0 or cpu")
    parser.add_argument("--imgsz", type=int, default=int(os.getenv("ROAD_IMGSZ", 640)))
    parser.add_argument("--host", default=os.getenv("ROAD_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("ROAD_PORT", 5000)))
    return parser.parse_args()


def main(opt):
    if not Path(opt.weights).exists():
        raise FileNotFoundError(f"权重不存在: {opt.weights}")
    print(f"加载模型: {opt.weights}", flush=True)
    init_detector(opt.weights, opt.device, opt.imgsz)
    app.run(host=opt.host, port=opt.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main(parse_opt())
