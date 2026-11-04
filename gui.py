# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""道路病害检测 PyQt5 前端。

启动时自动拉起 app.py（若本机后端未运行），支持图片、视频文件和摄像头的
逐帧检测，并在右侧展示检测框与结果明细。

启动:
    python gui.py
"""

import base64
import io
import os
import sys
from collections import deque
from pathlib import Path
from urllib.parse import urlparse

import cv2
import requests
from PIL import Image, ImageOps
from PyQt5.QtCore import QObject, QProcess, QProcessEnvironment, QRunnable, QSize, Qt, QThreadPool, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

APP_DIR = Path(__file__).resolve().parent
API_URL = os.getenv("ROAD_API_URL", "http://127.0.0.1:5000")


class BackendService(QObject):
    """检查并（必要时）启动本机 Flask 后端。"""

    ready = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, base_url):
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.trust_env = False  # 本机请求绕过系统代理
        self.proc = None
        self.logs = deque(maxlen=40)
        self.attempts = 0
        self.timer = QTimer(self)
        self.timer.setInterval(500)
        self.timer.timeout.connect(self._poll)

    def start(self):
        payload = self._health()
        if payload:
            self.ready.emit(payload)
            return
        if self.proc is not None and self.proc.state() != QProcess.NotRunning:
            self.attempts = 0
            self.timer.start()
            return

        parsed = urlparse(self.base_url)
        host, port = parsed.hostname, parsed.port or 5000
        if host not in ("127.0.0.1", "localhost"):
            self.failed.emit(f"无法连接远程后端: {self.base_url}")
            return

        self.proc = QProcess(self)
        self.proc.setWorkingDirectory(str(APP_DIR))
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._read_log)
        self.proc.errorOccurred.connect(self._on_process_error)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONIOENCODING", "utf-8")
        self.proc.setProcessEnvironment(env)

        args = [str(APP_DIR / "app.py"), "--host", host, "--port", str(port)]
        if os.getenv("ROAD_WEIGHTS"):
            args += ["--weights", os.environ["ROAD_WEIGHTS"]]
        if os.getenv("ROAD_DEVICE"):
            args += ["--device", os.environ["ROAD_DEVICE"]]
        self.attempts = 0
        self.proc.start(sys.executable, args)
        self.timer.start()

    def _health(self):
        try:
            response = self.session.get(f"{self.base_url}/api/health", timeout=1.5)
            if response.status_code == 200:
                payload = response.json()
                if payload.get("status") == "ok":
                    return payload
        except (requests.RequestException, ValueError):
            pass
        return None

    def _read_log(self):
        text = self.proc.readAllStandardOutput().data().decode("utf-8", "ignore")
        for line in text.splitlines():
            if line.strip():
                self.logs.append(line.strip())

    def _on_process_error(self, _):
        if self.timer.isActive():
            self.timer.stop()
            self.failed.emit("无法启动后端进程:\n" + "\n".join(self.logs[-8:]))

    def _poll(self):
        self.attempts += 1
        payload = self._health()
        if payload:
            self.timer.stop()
            self.ready.emit(payload)
            return
        if self.attempts > 3 and self.proc is not None and self.proc.state() == QProcess.NotRunning:
            self.timer.stop()
            self.failed.emit("后端启动失败:\n" + ("\n".join(self.logs[-8:]) or "进程已退出"))
            return
        if self.attempts >= 240:  # 约 2 分钟，覆盖首次加载模型
            self.timer.stop()
            self.failed.emit("等待后端就绪超时，请检查模型权重或手动运行 app.py")

    def stop(self):
        self.timer.stop()
        if self.proc is not None and self.proc.state() != QProcess.NotRunning:
            self.proc.terminate()
            if not self.proc.waitForFinished(3000):
                self.proc.kill()
                self.proc.waitForFinished(1000)


class DetectorApi:
    """调用后端检测接口。"""

    def __init__(self, base_url):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.trust_env = False

    def detect(self, jpeg_bytes, conf, iou):
        response = self.session.post(
            f"{self.base_url}/api/detect",
            files={"image": ("frame.jpg", jpeg_bytes, "image/jpeg")},
            data={"conf": f"{conf:.2f}", "iou": f"{iou:.2f}"},
            timeout=60,
        )
        try:
            payload = response.json()
        except ValueError:
            raise RuntimeError(f"后端返回异常（HTTP {response.status_code}）")
        if response.status_code != 200 or not payload.get("success"):
            raise RuntimeError(payload.get("error", f"HTTP {response.status_code}"))
        return payload


class DetectSignals(QObject):
    done = pyqtSignal(object, int)
    error = pyqtSignal(str, int)


class DetectTask(QRunnable):
    """在 QThreadPool 中执行一次 HTTP 检测，避免阻塞界面。"""

    def __init__(self, api, jpeg_bytes, conf, iou, generation):
        super().__init__()
        self.api = api
        self.jpeg_bytes = jpeg_bytes
        self.conf = conf
        self.iou = iou
        self.generation = generation
        self.signals = DetectSignals()

    def run(self):
        try:
            self.signals.done.emit(self.api.detect(self.jpeg_bytes, self.conf, self.iou), self.generation)
        except Exception as e:
            self.signals.error.emit(str(e), self.generation)


class PreviewLabel(QLabel):
    """按窗口尺寸等比缩放显示检测画面。"""

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(640, 360)
        self.setStyleSheet("background:#161616;color:#9a9a9a;border:1px solid #333333;")
        self._pixmap = None
        self.setText("请选择图片 / 视频，或开启摄像头")

    def set_image(self, pixmap):
        if pixmap is None or pixmap.isNull():
            return
        self._pixmap = pixmap
        self.setText("")
        self._render()

    def clear(self, text="请选择图片 / 视频，或开启摄像头"):
        self._pixmap = None
        self.setText(text)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render()

    def _render(self):
        if self._pixmap is not None:
            self.setPixmap(
                self._pixmap.scaled(self.size() - QSize(8, 8), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )


class MainWindow(QMainWindow):
    def __init__(self, api_url):
        super().__init__()
        self.setWindowTitle("道路病害检测系统")
        self.resize(1280, 760)
        self.api = DetectorApi(api_url)
        self.backend = BackendService(api_url)
        self.backend.ready.connect(self._on_backend_ready)
        self.backend.failed.connect(self._on_backend_failed)

        self.generation = 0
        self.busy = False
        self.ready = False
        self.mode = None  # image / video / camera
        self.image_bytes = None
        self.cap = None
        self.capture_timer = QTimer(self)
        self.capture_timer.timeout.connect(self._read_frame)

        self._build_ui()
        self._update_buttons()
        self.backend.start()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # 左侧控制面板
        left = QWidget()
        left.setFixedWidth(270)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        source_group = QGroupBox("输入源")
        source_layout = QVBoxLayout(source_group)
        self.btn_image = QPushButton("选择图片")
        self.btn_video = QPushButton("选择视频")
        self.btn_camera = QPushButton("开启摄像头")
        self.btn_stop = QPushButton("停止检测")
        self.btn_retry = QPushButton("重试连接后端")
        for button in (self.btn_image, self.btn_video, self.btn_camera, self.btn_stop):
            button.setMinimumHeight(34)
            source_layout.addWidget(button)
        source_layout.addWidget(self.btn_retry)
        left_layout.addWidget(source_group)

        param_group = QGroupBox("检测参数")
        param_layout = QVBoxLayout(param_group)
        self.conf_value = QLabel("置信度阈值  0.25")
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(1, 99)
        self.conf_slider.setValue(25)
        self.conf_slider.valueChanged.connect(lambda v: self.conf_value.setText(f"置信度阈值  {v / 100:.2f}"))
        self.iou_value = QLabel("NMS IoU  0.45")
        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setRange(1, 99)
        self.iou_slider.setValue(45)
        self.iou_slider.valueChanged.connect(lambda v: self.iou_value.setText(f"NMS IoU  {v / 100:.2f}"))
        self.btn_redetect = QPushButton("重新检测")
        self.btn_redetect.setMinimumHeight(34)
        param_layout.addWidget(self.conf_value)
        param_layout.addWidget(self.conf_slider)
        param_layout.addWidget(self.iou_value)
        param_layout.addWidget(self.iou_slider)
        param_layout.addWidget(self.btn_redetect)
        left_layout.addWidget(param_group)

        tips = QLabel("图片支持 JPG/PNG/BMP；\n视频支持 MP4/AVI/MOV/MKV；\n摄像头默认索引 0。")
        tips.setWordWrap(True)
        tips.setStyleSheet("color:#777777;")
        left_layout.addWidget(tips)
        left_layout.addStretch(1)
        layout.addWidget(left)

        # 中间预览
        self.preview = PreviewLabel()
        layout.addWidget(self.preview, 1)

        # 右侧结果
        right = QWidget()
        right.setFixedWidth(340)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        title = QLabel("检测结果")
        title.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        self.summary = QLabel("等待检测…")
        self.summary.setWordWrap(True)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["类别", "置信度", "位置 (x1,y1,x2,y2)"])
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(0, 110)
        self.table.setColumnWidth(1, 70)
        self.latency = QLabel("")
        self.latency.setWordWrap(True)
        self.latency.setStyleSheet("color:#777777;")
        right_layout.addWidget(title)
        right_layout.addWidget(self.summary)
        right_layout.addWidget(self.table, 1)
        right_layout.addWidget(self.latency)
        layout.addWidget(right)

        self.setCentralWidget(central)

        self.status_backend = QLabel("后端：启动中…")
        self.status_model = QLabel("")
        self.statusBar().addPermanentWidget(self.status_backend)
        self.statusBar().addPermanentWidget(self.status_model)
        self.statusBar().showMessage("正在启动后端并加载模型，请稍候…")

        self.btn_image.clicked.connect(self._choose_image)
        self.btn_video.clicked.connect(self._choose_video)
        self.btn_camera.clicked.connect(self._start_camera)
        self.btn_stop.clicked.connect(self._stop_capture)
        self.btn_retry.clicked.connect(self._retry_backend)
        self.btn_redetect.clicked.connect(self._redetect)

    # -------------------------------------------------------------- backend
    def _on_backend_ready(self, payload):
        model = payload.get("model", {})
        self.ready = True
        self.status_backend.setText("后端：就绪")
        self.status_model.setText(f"{model.get('device', '')} · {model.get('imgsz', '')}px")
        self.statusBar().showMessage(f"后端就绪，模型加载完成（{len(model.get('names', []))} 类）")
        self._update_buttons()

    def _on_backend_failed(self, message):
        self.ready = False
        self.status_backend.setText("后端：不可用")
        self._update_buttons()
        QMessageBox.critical(self, "后端启动失败", message)

    def _retry_backend(self):
        self.statusBar().showMessage("正在重新连接后端…")
        self.btn_retry.setEnabled(False)
        self.backend.start()

    # --------------------------------------------------------------- inputs
    def _choose_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", str(APP_DIR), "图片 (*.jpg *.jpeg *.png *.bmp *.webp)"
        )
        if path:
            self._start_image(path)

    def _choose_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", str(APP_DIR), "视频 (*.mp4 *.avi *.mov *.mkv *.wmv)"
        )
        if path:
            self._start_capture(path)

    def _start_camera(self):
        self._start_capture(0)

    def _start_image(self, path):
        self._stop_capture()
        try:
            with Image.open(path) as raw:
                image = ImageOps.exif_transpose(raw).convert("RGB")
                if max(image.size) > 1920:
                    image.thumbnail((1920, 1920))
                buffer = io.BytesIO()
                image.save(buffer, "JPEG", quality=92)
                data = buffer.getvalue()
        except Exception as e:
            QMessageBox.warning(self, "图片读取失败", str(e))
            return

        self.mode = "image"
        self.image_bytes = data
        self.generation += 1
        self._clear_results()
        self.preview.set_image(self._pixmap_from_bytes(data))
        self.statusBar().showMessage(f"已加载图片：{Path(path).name}，正在检测…")
        self._update_buttons()
        self._submit(data)

    def _start_capture(self, source):
        self._stop_capture()
        self.mode = "video" if isinstance(source, str) else "camera"
        self.image_bytes = None
        capture = (
            cv2.VideoCapture(source, cv2.CAP_DSHOW)
            if self.mode == "camera" and os.name == "nt"
            else cv2.VideoCapture(source)
        )
        if not capture.isOpened():
            capture.release()
            self.mode = None
            QMessageBox.warning(self, "打开失败", f"无法打开输入源：{source}")
            return

        self.cap = capture
        self.generation += 1
        self._clear_results()
        self.preview.clear("正在读取视频流…")
        interval = 33
        if self.mode == "video":
            fps = capture.get(cv2.CAP_PROP_FPS)
            if fps and fps > 0:
                interval = int(max(15, min(100, 1000 / fps)))
            self.statusBar().showMessage(f"正在检测视频：{Path(source).name}")
        else:
            self.statusBar().showMessage("摄像头检测中…")
        self.capture_timer.start(interval)
        self._update_buttons()

    def _read_frame(self):
        if self.cap is None:
            return
        ok, frame = self.cap.read()
        if not ok:
            if self.mode == "video":
                self._stop_capture()
                self.statusBar().showMessage("视频检测结束")
            elif self.mode == "camera":
                self._stop_capture()
                self.statusBar().showMessage("摄像头读取失败，已停止")
            return
        if self.busy:
            return
        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            self._submit(buffer.tobytes())

    def _stop_capture(self):
        self.capture_timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.mode in ("video", "camera"):
            self.mode = "image" if self.image_bytes else None
            self.generation += 1
        self._update_buttons()

    def _redetect(self):
        if self.image_bytes:
            self.generation += 1
            self._clear_results()
            self._submit(self.image_bytes)

    # ------------------------------------------------------------ detection
    def _submit(self, jpeg_bytes):
        if self.busy or not self.ready:
            return
        self.busy = True
        self._update_buttons()
        task = DetectTask(
            self.api,
            jpeg_bytes,
            self.conf_slider.value() / 100,
            self.iou_slider.value() / 100,
            self.generation,
        )
        task.signals.done.connect(self._on_result)
        task.signals.error.connect(self._on_error)
        QThreadPool.globalInstance().start(task)

    def _on_result(self, payload, generation):
        self.busy = False
        self._update_buttons()
        if generation != self.generation:
            # 上一次请求返回时输入已切换，若当前是图片模式则补发当前图片
            if self.mode == "image" and self.image_bytes:
                self._submit(self.image_bytes)
            return

        pixmap = self._pixmap_from_data_uri(payload.get("annotated", ""))
        if pixmap is not None:
            self.preview.set_image(pixmap)

        detections = payload.get("detections", [])
        counts = payload.get("counts", {})
        self.table.setRowCount(len(detections))
        for row, det in enumerate(detections):
            self.table.setItem(row, 0, QTableWidgetItem(det["class_name"]))
            confidence = QTableWidgetItem(f"{det['confidence']:.2f}")
            confidence.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 1, confidence)
            box = QTableWidgetItem(", ".join(str(v) for v in det["bbox"]))
            self.table.setItem(row, 2, box)

        if detections:
            lines = [f"共检测到 {len(detections)} 处目标"]
            lines += [f"{name}: {count}" for name, count in counts.items()]
            self.summary.setText("\n".join(lines))
        else:
            self.summary.setText("未检测到道路病害")

        elapsed = payload.get("elapsed_ms", {})
        self.latency.setText(
            f"耗时 {elapsed.get('total_ms', 0):.1f} ms"
            f"（预处理 {elapsed.get('preprocess_ms', 0):.1f} / "
            f"推理 {elapsed.get('inference_ms', 0):.1f} / "
            f"NMS {elapsed.get('nms_ms', 0):.1f}）"
        )

    def _on_error(self, message, generation):
        self.busy = False
        if generation != self.generation:
            return
        if self.mode in ("video", "camera"):
            self._stop_capture()
            self.statusBar().showMessage(f"检测失败：{message}")
            QMessageBox.warning(self, "检测失败", message)
        else:
            self._update_buttons()
            QMessageBox.warning(self, "检测失败", message)

    def _clear_results(self):
        self.table.setRowCount(0)
        self.summary.setText("等待检测…")
        self.latency.setText("")

    # --------------------------------------------------------------- helpers
    def _update_buttons(self):
        for button in (self.btn_image, self.btn_video, self.btn_camera):
            button.setEnabled(self.ready)
        self.btn_stop.setEnabled(self.mode in ("video", "camera"))
        self.btn_redetect.setEnabled(
            self.ready and self.mode == "image" and self.image_bytes is not None and not self.busy
        )
        self.btn_retry.setEnabled(not self.ready)

    @staticmethod
    def _pixmap_from_bytes(data):
        image = QImage()
        if image.loadFromData(data):
            return QPixmap(image)
        return None

    @staticmethod
    def _pixmap_from_data_uri(uri):
        if not uri:
            return None
        encoded = uri.split(",", 1)[1] if "," in uri else uri
        try:
            return MainWindow._pixmap_from_bytes(base64.b64decode(encoded))
        except (ValueError, TypeError):
            return None

    def closeEvent(self, event):
        self._stop_capture()
        QThreadPool.globalInstance().waitForDone(3000)
        self.backend.stop()
        event.accept()


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 9))
    window = MainWindow(API_URL)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
