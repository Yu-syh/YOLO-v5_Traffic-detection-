import requests
from pathlib import Path

urls = [
    "https://raw.githubusercontent.com/sekilab/RoadDamageDetector/master/figures/rdd2022_example.jpg",
    "https://raw.githubusercontent.com/sekilab/RoadDamageDetector/master/figures/road_damage.jpg",
]

save = Path(r"E:\lanzhi\YOLO\yolov5-master\1.jpg")

session = requests.Session()
session.trust_env = False  # 关键：忽略系统代理

for u in urls:
    try:
        r = session.get(u, timeout=30)
        r.raise_for_status()
        save.write_bytes(r.content)
        print("saved:", save, "size:", save.stat().st_size)
        break
    except Exception as e:
        print("fail:", u, e)