#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Frontend de apresentacao — Detector de Placas (MobileNetV1 Grid + ESP32).

Roda o MESMO modelo embarcado (modelo_grid_224_alpha_0p5_int8.tflite) no PC via
LiteRT (~4 ms/inferencia) e a segmentacao em OpenCV. Serve uma pagina web local
e offline em http://localhost:8000 com as 3 modalidades exigidas no enunciado:
Imagem, Video e Tempo real.

Uso:
    python frontend_apresentacao.py
    python frontend_apresentacao.py 9000        # outra porta web

Dependencias: ai-edge-litert, opencv-python, numpy  (ja instaladas).
Fase atual: modo IMAGEM completo (deteccao + heatmap + segmentacao).
            Video e Tempo real entram nas proximas fases.
"""
import sys
import os
import io
import re
import json
import time
import glob
import base64
import threading
from urllib.parse import unquote
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import cv2

try:
    from PIL import Image, ImageOps
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    print("ERRO: ai-edge-litert nao instalado. Rode: python -m pip install ai-edge-litert")
    sys.exit(1)

try:
    import serial  # pyserial (modo Tempo real / ESP)
    _HAS_SERIAL = True
except ImportError:
    _HAS_SERIAL = False

# --------------------------------------------------------------- config
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)  # raiz do repositorio (frontend/ fica dentro dela)
MODEL_PATH = os.path.join(HERE, "modelo_grid_224_alpha_0p5_int8.tflite")
IMG_SIZE = 224
GRID = 7
WEB_PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
LIMIAR_PLACA = 0.95  # mesmo limiar do firmware

# --------------------------------------------------------------- modelo
_interp = Interpreter(model_path=MODEL_PATH)
_interp.allocate_tensors()
_IN = _interp.get_input_details()[0]
_OUT = _interp.get_output_details()[0]
_IN_SCALE, _IN_ZERO = _IN["quantization"]
_OUT_SCALE, _OUT_ZERO = _OUT["quantization"]
_infer_lock = threading.Lock()


# ------------------------------------------------------ decode (da notebook)
def pos_processar_bbox(bbox):
    bbox = np.array(bbox, dtype=np.float32).copy()
    bbox = np.nan_to_num(bbox, nan=0.0, posinf=1.0, neginf=0.0)
    bbox[0] = np.clip(bbox[0], 0.0, 1.0)
    bbox[1] = np.clip(bbox[1], 0.0, 1.0)
    bbox[2] = np.clip(bbox[2], 0.01, 1.0)
    bbox[3] = np.clip(bbox[3], 0.01, 1.0)
    return bbox


def yolo_to_xyxy_norm(box):
    box = pos_processar_bbox(box)
    cx, cy, w, h = box
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy + h / 2
    return np.array([np.clip(x1, 0, 1), np.clip(y1, 0, 1),
                     np.clip(x2, 0, 1), np.clip(y2, 0, 1)], dtype=np.float32)


def bbox_yolo_para_pixel(box, img_w, img_h):
    x1, y1, x2, y2 = yolo_to_xyxy_norm(box)
    x1, y1 = int(x1 * img_w), int(y1 * img_h)
    x2, y2 = int(x2 * img_w), int(y2 * img_h)
    x1 = max(0, min(x1, img_w - 2)); y1 = max(0, min(y1, img_h - 2))
    x2 = max(x1 + 1, min(x2, img_w - 1)); y2 = max(y1 + 1, min(y2, img_h - 1))
    return x1, y1, x2, y2


def grade_para_bbox_global(pred_grid, limiar_conf=0.0):
    confs = pred_grid[..., 0]
    row, col = np.unravel_index(np.argmax(confs), confs.shape)
    conf_max = float(confs[row, col])
    if conf_max < limiar_conf:
        return None, conf_max, (row, col)
    cx = (col + float(pred_grid[row, col, 1])) / GRID
    cy = (row + float(pred_grid[row, col, 2])) / GRID
    w = float(pred_grid[row, col, 3]); h = float(pred_grid[row, col, 4])
    bbox = np.array([np.clip(cx, 0, 1), np.clip(cy, 0, 1),
                     np.clip(w, 0.01, 1), np.clip(h, 0.01, 1)], dtype=np.float32)
    return bbox, conf_max, (int(row), int(col))


# ------------------------------------------------------ inferencia
def preprocess(img_bgr, modo="rgb"):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if modo == "gray3":
        g = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        img_rgb = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)
    resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    norm = (resized.astype(np.float32) / 127.5) - 1.0
    q = np.round(norm / _IN_SCALE + _IN_ZERO)
    q = np.clip(q, -128, 127).astype(np.int8)
    return q[np.newaxis, ...], resized


def detectar(img_bgr, modo="rgb"):
    q, vis = preprocess(img_bgr, modo)
    with _infer_lock:
        _interp.set_tensor(_IN["index"], q)
        _interp.invoke()
        out = _interp.get_tensor(_OUT["index"])[0]
    pred = (out.astype(np.float32) - _OUT_ZERO) * _OUT_SCALE  # (7,7,5) em [0,1]
    bbox, conf, cell = grade_para_bbox_global(pred, limiar_conf=0.0)
    return pred, bbox, conf, cell, vis


# ------------------------------------------------------ segmentacao (cv2, da notebook)
def expandir_bbox_pixel(x1, y1, x2, y2, img_w, img_h, padding=0.06):
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * padding), int(bh * padding)
    return (max(0, x1 - px), max(0, y1 - py),
            min(img_w - 1, x2 + px), min(img_h - 1, y2 + py))


def segmentar_roi_placa(img_rgb, bbox_yolo, padding=0.06, escala=4):
    img_h, img_w = img_rgb.shape[:2]
    x1, y1, x2, y2 = bbox_yolo_para_pixel(bbox_yolo, img_w, img_h)
    x1, y1, x2, y2 = expandir_bbox_pixel(x1, y1, x2, y2, img_w, img_h, padding)
    roi = img_rgb[y1:y2, x1:x2].copy()
    if roi.size == 0:
        z = np.zeros((10, 10), np.uint8)
        return roi, z, roi
    h0, w0 = roi.shape[:2]
    roi_up = cv2.resize(roi, (w0 * escala, h0 * escala), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(roi_up, cv2.COLOR_RGB2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4)).apply(gray)
    gray = cv2.bilateralFilter(gray, 5, 50, 50)
    h, w = gray.shape[:2]
    kb = cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, w // 6), max(3, h // 4)))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kb)
    _, m1 = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 21, 7)
    mask = cv2.bitwise_or(m1, m2)
    mx, my = int(0.04 * w), int(0.16 * h)
    mask[:my, :] = 0; mask[h - my:, :] = 0; mask[:, :mx] = 0; mask[:, w - mx:] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filt = np.zeros_like(mask); area_total = h * w
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        if x <= 1 or y <= 1 or x + ww >= w - 1 or y + hh >= h - 1:
            continue
        af = area / area_total
        if af < 0.001 or af > 0.20 or hh / h < 0.12 or ww / w > 0.45:
            continue
        filt[labels == i] = 255
    if np.sum(filt > 0) < 0.01 * area_total:
        filt = mask
    seg = cv2.bitwise_and(roi_up, roi_up, mask=filt)
    return roi_up, filt, seg


# ------------------------------------------------------ render helpers
def decode_image_bytes(raw):
    """Decodifica bytes de imagem em BGR, respeitando orientacao EXIF (fotos de celular)."""
    if _HAS_PIL:
        try:
            im = Image.open(io.BytesIO(raw))
            im = ImageOps.exif_transpose(im).convert("RGB")
            return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        except Exception:
            pass
    arr = np.frombuffer(raw, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def png_b64(img_rgb):
    if img_rgb.ndim == 2:
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2BGR)
    else:
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    return "data:image/png;base64," + base64.b64encode(buf).decode()


def heatmap_rgb(pred):
    cm = (np.clip(pred[..., 0], 0, 1) * 255).astype(np.uint8)
    big = cv2.resize(cm, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    col = cv2.applyColorMap(big, cv2.COLORMAP_JET)
    return cv2.cvtColor(col, cv2.COLOR_BGR2RGB)


def processar_imagem(img_bgr, modo="rgb"):
    pred, bbox, conf, cell, vis = detectar(img_bgr, modo)
    img_box = vis.copy()
    cor = (0, 200, 0) if conf >= LIMIAR_PLACA else (255, 140, 0)
    x1, y1, x2, y2 = bbox_yolo_para_pixel(bbox, IMG_SIZE, IMG_SIZE)
    cv2.rectangle(img_box, (x1, y1), (x2, y2), cor, 2)
    # segmentacao na imagem original (full-res) para caracteres nitidos
    img_rgb_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    roi_up, mask, seg = segmentar_roi_placa(img_rgb_full, bbox)
    return {
        "conf": round(conf, 4),
        "placa": bool(conf >= LIMIAR_PLACA),
        "cell": list(cell),
        "modo": modo,
        "img_original": png_b64(vis),
        "img_bbox": png_b64(img_box),
        "img_heatmap": png_b64(heatmap_rgb(pred)),
        "img_roi": png_b64(roi_up),
        "img_mask": png_b64(mask),
        "img_seg": png_b64(seg),
    }


# ------------------------------------------------------ video (modo Video)
def listar_videos():
    exts = ("*.mp4", "*.mov", "*.avi", "*.mkv", "*.webm")
    achados = []
    for e in exts:
        achados += glob.glob(os.path.join(REPO, "**", e), recursive=True)
    rel = sorted(os.path.relpath(p, REPO).replace("\\", "/") for p in achados)
    return rel


def anotar_frame(frame_bgr, modo="rgb", fps_txt=""):
    pred, bbox, conf, cell, _ = detectar(frame_bgr, modo)
    h, w = frame_bgr.shape[:2]
    out = frame_bgr.copy()
    placa = conf >= LIMIAR_PLACA
    cor = (0, 200, 0) if placa else (0, 140, 255)  # BGR
    x1, y1, x2, y2 = bbox_yolo_para_pixel(bbox, w, h)
    cv2.rectangle(out, (x1, y1), (x2, y2), cor, 3)
    tag = "PLACA" if placa else "sem placa"
    txt = f"conf={conf:.2f}  {tag}  {fps_txt}"
    cv2.rectangle(out, (0, 0), (w, 34), (0, 0, 0), -1)
    cv2.putText(out, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)
    return out, conf, placa


def gerar_stream_video(path_rel, modo="rgb"):
    full = os.path.normpath(os.path.join(REPO, path_rel))
    if not full.startswith(REPO) or not os.path.isfile(full):
        return
    cap = cv2.VideoCapture(full)
    vid_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    alvo_dt = 1.0 / min(max(vid_fps, 1), 30)
    while True:
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop
            continue
        t0 = time.time()
        try:
            out, conf, placa = anotar_frame(frame, modo, "")
            fps_inst = 1.0 / max(time.time() - t0, 1e-3)
            cv2.putText(out, f"FPS~{fps_inst:.0f}", (8, out.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        except Exception:
            out = frame
        ok2, jpg = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok2:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
               + str(len(jpg)).encode() + b"\r\n\r\n" + jpg.tobytes() + b"\r\n")
        dt = time.time() - t0
        if dt < alvo_dt:
            time.sleep(alvo_dt - dt)
    cap.release()


# ------------------------------------------------------ ESP (modo Tempo real)
ESP_THR = 0.95
esp_lock = threading.Lock()
esp_state = {
    "connected": False, "board_status": "off", "port": "",
    "confidence": None, "frames_consec": 0, "frames_needed": 2,
    "threshold": ESP_THR, "infer_est": 117,
    "history": [], "events": [], "counters": {"frames": 0, "detections": 0},
    "last_frame_epoch": None,
}
_esp_thread = None
_RE_CONF = re.compile(r"grelha:\s*([0-9]*\.?[0-9]+)")
_RE_FR = re.compile(r"consecutivos com placa:\s*(\d+)")


def _esp_event(msg, kind="info"):
    esp_state["events"].insert(0, {"t": time.strftime("%H:%M:%S"), "msg": msg, "kind": kind})
    del esp_state["events"][40:]


def _esp_line(line):
    with esp_lock:
        if "Wi-Fi Ligado" in line:
            esp_state["board_status"] = "wifi"; _esp_event("WiFi conectado", "ok")
        elif "PSRAM" in line and "Dete" in line:
            _esp_event("PSRAM detectada", "ok")
        elif "Pipeline pronto" in line:
            esp_state["board_status"] = "ready"; _esp_event("Pipeline pronto", "ok")
        elif "Placa dete" in line:
            esp_state["counters"]["detections"] += 1
            _esp_event("PLACA DETECTADA — bounding box", "detect")
        elif "SUCESSO" in line and "enviada" in line:
            _esp_event("Foto enviada ao Telegram", "sent")
        else:
            m = _RE_CONF.search(line)
            if m:
                c = float(m.group(1)); now = time.time()
                esp_state["confidence"] = c; esp_state["last_frame_epoch"] = now
                esp_state["counters"]["frames"] += 1
                esp_state["history"].append({"t": now, "c": c})
                del esp_state["history"][:-120]
                _esp_event(f"Frame: confianca {c:.2f}", "detect" if c >= ESP_THR else "info")
            mf = _RE_FR.search(line)
            if mf:
                esp_state["frames_consec"] = int(mf.group(1))


def _esp_worker(port):
    while True:
        try:
            ser = serial.Serial(port, 115200, timeout=2)
            try:
                ser.dtr = False; ser.rts = True; time.sleep(0.2); ser.rts = False
            except Exception:
                pass
            with esp_lock:
                esp_state["connected"] = True; esp_state["port"] = port
                esp_state["board_status"] = "init"
                _esp_event(f"Serial {port} aberta — aguardando boot", "ok")
            buf = b""
            while True:
                chunk = ser.read(256)
                if chunk:
                    buf += chunk
                    while b"\n" in buf:
                        raw, buf = buf.split(b"\n", 1)
                        s = raw.decode("utf-8", errors="ignore").strip()
                        if s:
                            _esp_line(s)
        except Exception as e:
            with esp_lock:
                esp_state["connected"] = False
                _esp_event(f"Serial indisponivel ({e}); tentando...", "warn")
            time.sleep(3)


def esp_start(port):
    global _esp_thread
    if not _HAS_SERIAL:
        return False, "pyserial nao instalado"
    if _esp_thread and _esp_thread.is_alive():
        return True, "ja conectado"
    _esp_thread = threading.Thread(target=_esp_worker, args=(port,), daemon=True)
    _esp_thread.start()
    return True, "conectando"


# --------------------------------------------------------------- pagina
PAGE = r"""<!DOCTYPE html><html lang="pt-br"><head><meta charset="utf-8">
<title>Detector de Placas — Apresentacao</title>
<style>
*{box-sizing:border-box} body{margin:0;font-family:'Segoe UI',system-ui,sans-serif;
background:#0d1117;color:#e6edf3}
header{padding:16px 26px;background:linear-gradient(90deg,#161b22,#1f2937);
border-bottom:1px solid #30363d}
header h1{margin:0;font-size:19px} header .s{color:#8b949e;font-size:13px;margin-top:2px}
.tabs{display:flex;gap:6px;padding:0 26px;background:#161b22;border-bottom:1px solid #30363d}
.tab{padding:12px 18px;cursor:pointer;color:#8b949e;border-bottom:2px solid transparent}
.tab.on{color:#58a6ff;border-bottom-color:#1f6feb;font-weight:600}
main{padding:22px 26px;max-width:1150px;margin:0 auto}
.panel{display:none} .panel.on{display:block}
.row{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin-bottom:16px}
button,select,label.file{background:#21262d;color:#e6edf3;border:1px solid #30363d;
border-radius:8px;padding:9px 16px;cursor:pointer;font-size:14px}
button.primary{background:#1f6feb;border-color:#1f6feb;font-weight:600}
.badge{padding:8px 16px;border-radius:20px;font-weight:700;font-size:15px}
.b-plate{background:#0f3d22;color:#3fb950;border:1px solid #238636}
.b-no{background:#21262d;color:#8b949e;border:1px solid #30363d}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}
@media(max-width:820px){.grid{grid-template-columns:1fr 1fr}}
.card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:10px}
.card h3{margin:0 0 8px;font-size:12px;color:#8b949e;text-transform:uppercase;letter-spacing:.5px}
.card img{width:100%;border-radius:6px;display:block;background:#0d1117}
.muted{color:#8b949e;font-size:13px} .soon{padding:40px;text-align:center;color:#8b949e}
.mcard{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:14px 18px;margin-bottom:16px}
.mcard h3{margin:0 0 10px;font-size:14px;color:#e6edf3}
table.mt{width:100%;border-collapse:collapse;font-size:13px}
table.mt th,table.mt td{padding:8px 10px;border-bottom:1px solid #21262d;text-align:left}
table.mt th{color:#8b949e;font-weight:600} table.mt td.ok{color:#3fb950;font-weight:600}
</style></head><body>
<header><h1>Detector de Placas Veiculares — MobileNetV1 Grid + ESP32</h1>
<div class="s">Mesmo modelo embarcado rodando no PC via LiteRT (~4 ms) + segmentacao OpenCV</div></header>
<div class="tabs">
  <div class="tab on" data-t="img">Imagem</div>
  <div class="tab" data-t="vid">Video</div>
  <div class="tab" data-t="rt">Tempo real</div>
  <div class="tab" data-t="met">Metricas</div>
</div>
<main>
  <div id="p-img" class="panel on">
    <div class="row">
      <label class="file">Escolher imagem<input id="file" type="file" accept="image/*" hidden></label>
      <select id="modo"><option value="rgb">Modo RGB</option><option value="gray3">Modo gray3</option></select>
      <button class="primary" id="run">Detectar</button>
      <span id="status" class="muted"></span>
    </div>
    <div class="row"><span id="badge" class="badge b-no">aguardando imagem</span>
      <span id="info" class="muted"></span></div>
    <div class="grid">
      <div class="card"><h3>Entrada 224x224</h3><img id="i-original"></div>
      <div class="card"><h3>Deteccao (bbox)</h3><img id="i-bbox"></div>
      <div class="card"><h3>Mapa de confianca (grade 7x7)</h3><img id="i-heatmap"></div>
      <div class="card"><h3>ROI da placa</h3><img id="i-roi"></div>
      <div class="card"><h3>Mascara (segmentacao)</h3><img id="i-mask"></div>
      <div class="card"><h3>Caracteres segmentados</h3><img id="i-seg"></div>
    </div>
  </div>
  <div id="p-vid" class="panel">
    <div class="row">
      <select id="vsel" style="min-width:280px"></select>
      <select id="vmodo"><option value="rgb">Modo RGB</option><option value="gray3">Modo gray3</option></select>
      <button class="primary" id="vstart">Iniciar</button>
      <button id="vstop">Parar</button>
      <span id="vstatus" class="muted"></span>
    </div>
    <div class="card" style="max-width:520px"><h3>Deteccao em video (ao vivo)</h3>
      <img id="vstream" style="width:100%;border-radius:6px;background:#0d1117;min-height:200px"></div>
    <p class="muted">A bbox aparece sobre o video rodando. Pause/print quando a placa for detectada (caixa verde).</p>
  </div>
  <div id="p-rt" class="panel">
    <div class="row">
      <input id="rtport" value="COM3" style="width:90px;background:#21262d;color:#e6edf3;border:1px solid #30363d;border-radius:8px;padding:9px">
      <button class="primary" id="rtconn">Conectar a ESP</button>
      <span id="rtconnstatus" class="muted"></span>
    </div>
    <p class="muted">Filmagem em tempo real com a camera OV5640 acoplada a ESP32-S3 (inferencia embarcada, ~2 min/frame). A ESP envia a foto com bounding box ao Telegram quando confianca &ge; 0,95.</p>
    <div class="row">
      <span id="rtbadge" class="badge b-no">ESP desconectada</span>
      <span id="rtinfo" class="muted"></span>
    </div>
    <div id="rtsub" class="muted" style="margin:-6px 0 12px">—</div>
    <div style="margin-bottom:16px">
      <div style="display:flex;justify-content:space-between;font-size:12px;color:#8b949e;margin-bottom:4px">
        <span id="rtprogtxt">Inferencia do frame atual</span><span id="rtprogpct"></span></div>
      <div style="height:10px;background:#21262d;border-radius:6px;overflow:hidden;border:1px solid #30363d">
        <div id="rtprog" style="height:100%;width:0%;background:#1f6feb;transition:width 1s linear"></div></div>
    </div>
    <div class="grid" style="grid-template-columns:1.3fr 1fr">
      <div class="mcard">
        <h3>Confianca (Y) ao longo do tempo (X) — limiar 0,95</h3>
        <svg id="rtchart" viewBox="0 0 620 250" style="width:100%;height:250px"></svg>
        <div class="muted" style="font-size:12px">● ponto = um frame analisado pela ESP &nbsp;·&nbsp; <span style="color:#3fb950">verde</span> ≥ 0,95 (placa) &nbsp;·&nbsp; <span style="color:#58a6ff">azul</span> abaixo</div>
      </div>
      <div class="mcard">
        <h3>Eventos da ESP</h3>
        <div id="rtlog" style="max-height:200px;overflow:auto;font-size:13px"></div>
      </div>
    </div>
    <div class="row">
      <span class="muted">Frames analisados: <b id="rtfr">0</b></span>
      <span class="muted">Deteccoes enviadas: <b id="rtdet">0</b></span>
      <span class="muted">~117 s por inferencia (embarcado)</span>
    </div>
  </div>
  <div id="p-met" class="panel">
    <p class="muted">Resultados reais medidos no projeto (treino no Colab + execucao na ESP32-S3). Use estes paineis para prints do relatorio.</p>
    <div class="mcard">
      <h3>Deteccao por grade — 5 execucoes (media &plusmn; desvio)</h3>
      <table class="mt">
        <tr><th>Metrica</th><th>Base 1 (prof.)</th><th>Base 2 (escolhida)</th><th>Mix</th></tr>
        <tr><td>IoU medio</td><td>0,7682 &plusmn; 0,0024</td><td>0,5427 &plusmn; 0,0304</td><td>0,7306 &plusmn; 0,0048</td></tr>
        <tr><td>Recall (IoU&ge;0,5)</td><td>0,9733 &plusmn; 0,0133</td><td>0,6667 &plusmn; 0,0745</td><td>0,9222 &plusmn; 0,0142</td></tr>
        <tr><td>F1@0,5</td><td>0,983</td><td>0,750</td><td>0,944</td></tr>
      </table>
      <p class="muted">Score balanceado: 0,6554 &plusmn; 0,0148. Melhor modelo = execucao 5.</p>
    </div>
    <div class="mcard">
      <h3>Comparacao de plataformas — tempo de inferencia &amp; tamanho</h3>
      <table class="mt">
        <tr><th>Modelo / plataforma</th><th>Tempo/inferencia</th><th>Tamanho</th><th>IoU (Mix)</th></tr>
        <tr><td>Keras (PC, float)</td><td>~96,9 ms</td><td>—</td><td>0,7362</td></tr>
        <tr><td>TFLite Float32 (PC)</td><td>~6,5 ms</td><td>8,75 MB</td><td>0,7362</td></tr>
        <tr><td>TFLite INT8 (PC, LiteRT)</td><td><b>~4&ndash;7 ms</b></td><td>2,35 MB</td><td>0,7321</td></tr>
        <tr><td><b>TFLite INT8 (ESP32-S3)</b></td><td><b>~117.000 ms</b></td><td>2,35 MB</td><td>—</td></tr>
      </table>
      <p class="muted">Quantizacao preservou a qualidade (IoU ~igual). PC e ~17.000&times; mais rapido que a ESP.</p>
    </div>
    <div class="mcard">
      <h3>Efeito das negativas — falsos positivos por limiar (4 imagens sem placa)</h3>
      <table class="mt">
        <tr><th>Limiar</th><th>0,50</th><th>0,90</th><th>0,95</th><th>0,98</th></tr>
        <tr><td>Falsos positivos</td><td>4/4</td><td>1/4</td><td><b>0/4</b></td><td>0/4</td></tr>
      </table>
      <p class="muted">Confirma o limiar 0,95: zero falso positivo, sem degradar os positivos (conf media ~0,99).</p>
    </div>
    <div class="mcard">
      <h3>Confianca medida na ESP32-S3 (modelo antigo &times; novo com negativas)</h3>
      <table class="mt">
        <tr><th>Cenario</th><th>Antigo</th><th>Novo (negativas)</th><th>Dispara?</th></tr>
        <tr><td>Placa real, camera correta</td><td>0,91&ndash;0,97</td><td><b>0,98&ndash;0,99</b></td><td class="ok">Sim</td></tr>
        <tr><td>Placa numa tela de monitor</td><td>0,62&ndash;0,83</td><td>0,95&ndash;0,98</td><td class="ok">Sim</td></tr>
        <tr><td>Site/tela sem carros</td><td>~0,62</td><td>0,63&ndash;0,78</td><td>Nao</td></tr>
        <tr><td>Sem placa nenhuma</td><td>~0,62</td><td><b>0,42&ndash;0,58</b></td><td>Nao</td></tr>
      </table>
    </div>
  </div>
</main>
<script>
document.querySelectorAll('.tab').forEach(t=>t.onclick=()=>{
  document.querySelectorAll('.tab').forEach(x=>x.classList.remove('on'));
  document.querySelectorAll('.panel').forEach(x=>x.classList.remove('on'));
  t.classList.add('on'); document.getElementById('p-'+t.dataset.t).classList.add('on');
});
let fileBytes=null, busy=false;
const PANELS=['original','bbox','heatmap','roi','mask','seg'];
function setStatus(t){document.getElementById('status').textContent=t;}
function clearPanels(){
  PANELS.forEach(k=>document.getElementById('i-'+k).removeAttribute('src'));
  const b=document.getElementById('badge'); b.className='badge b-no'; b.textContent='processando...';
  document.getElementById('info').textContent='';
}
async function loadFile(f){
  if(!f) return;
  fileBytes=new Uint8Array(await f.arrayBuffer());
  setStatus(f.name+' carregada');
  runDetect();           // auto-detecta ao escolher
}
async function runDetect(){
  if(!fileBytes){setStatus('escolha uma imagem primeiro');return;}
  if(busy) return; busy=true;
  const run=document.getElementById('run'); run.disabled=true;
  const modo=document.getElementById('modo').value;
  clearPanels(); setStatus('processando...');
  try{
    const r=await fetch('/detectar?modo='+modo,{method:'POST',body:fileBytes});
    const d=await r.json();
    if(d.erro){setStatus('erro: '+d.erro);}
    else{
      PANELS.forEach(k=>document.getElementById('i-'+k).src=d['img_'+k]);
      const b=document.getElementById('badge');
      if(d.placa){b.className='badge b-plate';b.textContent='PLACA DETECTADA';}
      else{b.className='badge b-no';b.textContent='SEM PLACA (abaixo de 0,95)';}
      document.getElementById('info').textContent=
        'confianca '+d.conf.toFixed(3).replace('.',',')+' · celula ['+d.cell+'] · modo '+d.modo;
      setStatus('ok');
    }
  }catch(err){setStatus('falha: '+err);}
  busy=false; run.disabled=false;
}
const fileInput=document.getElementById('file');
fileInput.onchange=e=>{ loadFile(e.target.files[0]); fileInput.value=''; };  // reset p/ reescolher mesma img
document.getElementById('run').onclick=runDetect;
document.getElementById('modo').onchange=()=>{ if(fileBytes) runDetect(); }; // re-roda ao trocar rgb/gray3
// arrastar e soltar
const drop=document.getElementById('p-img');
['dragover','dragenter'].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.style.outline='2px dashed #1f6feb';}));
['dragleave','drop'].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.style.outline='none';}));
drop.addEventListener('drop',e=>{ if(e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0]); });
// ----- modo Video -----
const vsel=document.getElementById('vsel');
async function carregarVideos(){
  try{
    const lista=await (await fetch('/videos')).json();
    vsel.innerHTML='';
    lista.forEach(p=>{const o=document.createElement('option');o.value=p;
      o.textContent=p.split('/').pop();vsel.appendChild(o);});
    const wa=lista.findIndex(p=>p.toLowerCase().includes('whatsapp'));
    if(wa>=0) vsel.selectedIndex=wa;
    document.getElementById('vstatus').textContent=lista.length+' video(s) encontrado(s)';
  }catch(e){document.getElementById('vstatus').textContent='erro ao listar videos';}
}
function vStart(){
  if(!vsel.value){document.getElementById('vstatus').textContent='nenhum video';return;}
  const modo=document.getElementById('vmodo').value;
  document.getElementById('vstream').src='/video_stream?modo='+modo+'&path='+encodeURIComponent(vsel.value)+'&t='+Date.now();
  document.getElementById('vstatus').textContent='rodando: '+vsel.value.split('/').pop();
}
function vStop(){document.getElementById('vstream').removeAttribute('src');
  document.getElementById('vstatus').textContent='parado';}
document.getElementById('vstart').onclick=vStart;
document.getElementById('vstop').onclick=vStop;
document.getElementById('vmodo').onchange=()=>{ if(document.getElementById('vstream').src) vStart(); };
carregarVideos();
// ----- modo Tempo real (ESP) -----
document.getElementById('rtconn').onclick=async ()=>{
  const port=document.getElementById('rtport').value||'COM3';
  document.getElementById('rtconnstatus').textContent='conectando...';
  try{const d=await (await fetch('/esp_connect?port='+encodeURIComponent(port),{method:'POST'})).json();
    document.getElementById('rtconnstatus').textContent=d.ok?('ok: '+d.msg):('erro: '+d.msg);
  }catch(e){document.getElementById('rtconnstatus').textContent='falha: '+e;}
};
function hhmmss(epoch){const d=new Date(epoch*1000);return d.toTimeString().slice(0,8);}
function rtChart(hist){
  const W=620,H=250,L=44,R=14,T=12,B=34;
  const pw=W-L-R, ph=H-T-B, x0=L, y0=T+ph;
  let s='';
  // eixos
  s+=`<line x1="${L}" y1="${T}" x2="${L}" y2="${y0}" stroke="#30363d" stroke-width="1"/>`;
  s+=`<line x1="${L}" y1="${y0}" x2="${W-R}" y2="${y0}" stroke="#30363d" stroke-width="1"/>`;
  // grade Y + rotulos (confianca 0..1)
  [0,0.25,0.5,0.75,1].forEach(v=>{const y=T+(1-v)*ph;
    s+=`<line x1="${L}" y1="${y}" x2="${W-R}" y2="${y}" stroke="#21262d" stroke-width="1"/>`;
    s+=`<text x="${L-6}" y="${y+4}" fill="#8b949e" font-size="11" text-anchor="end">${v.toFixed(2).replace('.',',')}</text>`;});
  // linha do limiar 0,95
  const yT=T+(1-0.95)*ph;
  s+=`<line x1="${L}" y1="${yT}" x2="${W-R}" y2="${yT}" stroke="#f0883e" stroke-width="1.5" stroke-dasharray="6 5"/>`;
  s+=`<text x="${W-R}" y="${yT-4}" fill="#f0883e" font-size="11" text-anchor="end">limiar 0,95</text>`;
  // titulos dos eixos
  s+=`<text transform="translate(13,${T+ph/2}) rotate(-90)" fill="#8b949e" font-size="12" text-anchor="middle">confianca</text>`;
  s+=`<text x="${W-R}" y="${H-6}" fill="#8b949e" font-size="12" text-anchor="end">tempo &#8594;</text>`;
  if(hist.length){
    const n=hist.length, st=n>1?pw/(n-1):0;
    const pts=hist.map((q,i)=>[x0+(n>1?i*st:pw/2), T+(1-Math.max(0,Math.min(1,q.c)))*ph, q.c, q.t]);
    s+=`<path d="${pts.map((q,i)=>(i?'L':'M')+q[0].toFixed(1)+' '+q[1].toFixed(1)).join(' ')}" fill="none" stroke="#58a6ff" stroke-width="2"/>`;
    const step=Math.max(1,Math.ceil(n/6));
    pts.forEach((q,i)=>{const col=q[2]>=0.95?'#3fb950':'#58a6ff';
      s+=`<circle cx="${q[0].toFixed(1)}" cy="${q[1].toFixed(1)}" r="3.5" fill="${col}"/>`;
      if(i%step===0||i===n-1)
        s+=`<text x="${q[0].toFixed(1)}" y="${y0+15}" fill="#8b949e" font-size="10" text-anchor="middle">${hhmmss(q[3])}</text>`;});
  }else{
    s+=`<text x="${L+pw/2}" y="${T+ph/2}" fill="#6e7681" font-size="13" text-anchor="middle">aguardando o 1o frame da ESP (~2 min)...</text>`;
  }
  document.getElementById('rtchart').innerHTML=s;
}
async function rtTick(){
  let d; try{d=await (await fetch('/esp_data')).json();}catch(e){return;}
  const b=document.getElementById('rtbadge'),c=d.confidence;
  let sub='';
  if(!d.connected){b.className='badge b-no';b.textContent='ESP desconectada';
    sub='Clique em "Conectar a ESP" para iniciar o monitoramento (porta COM3).';}
  else if(d.board_status==='init'||d.board_status==='wifi'){b.className='badge b-no';b.textContent='ESP inicializando';
    sub='A ESP esta ligando e conectando ao Wi-Fi. Verifique se o hotspot "4444" esta ligado.';}
  else if(c==null){b.className='badge b-no';b.textContent='Analisando 1o frame';
    sub='ESP capturando e processando o primeiro frame na propria placa (inferencia embarcada, ~2 min).';}
  else if(c>=0.95){b.className='badge b-plate';b.textContent='PLACA DETECTADA';
    sub='Confianca alta! Mantendo por '+d.frames_consec+'/'+d.frames_needed+' frames; ao confirmar, a ESP envia a foto com bounding box ao Telegram.';}
  else{b.className='badge b-no';b.textContent='SEM PLACA';
    sub='Analisando... ainda sem placa com confianca suficiente. Aponte a camera da ESP para uma placa bem enquadrada.';}
  document.getElementById('rtsub').textContent=sub;
  document.getElementById('rtinfo').textContent=
    (c==null?'':'confianca '+c.toFixed(2).replace('.',',')+' · ')+
    'frames consec. '+d.frames_consec+'/'+d.frames_needed;
  document.getElementById('rtfr').textContent=d.counters.frames;
  document.getElementById('rtdet').textContent=d.counters.detections;
  // barra de progresso da inferencia (~117 s)
  const prog=document.getElementById('rtprog'), ptxt=document.getElementById('rtprogtxt'), ppct=document.getElementById('rtprogpct');
  if(d.connected && d.board_status==='ready' && d.last_frame_epoch){
    const el=Date.now()/1000-d.last_frame_epoch, pct=Math.min(100, el/d.infer_est*100);
    prog.style.width=pct+'%'; ptxt.textContent='Analisando frame (embarcado)';
    ppct.textContent=Math.round(el)+'s / ~'+d.infer_est+'s';
  } else if(d.connected && d.board_status==='ready'){
    prog.style.width='15%'; ptxt.textContent='Primeiro frame em processamento'; ppct.textContent='~2 min';
  } else { prog.style.width='0%'; ptxt.textContent='Inferencia do frame atual'; ppct.textContent=''; }
  document.getElementById('rtlog').innerHTML=d.events.map(e=>
    `<div style="padding:5px 6px;border-bottom:1px solid #21262d"><span style="color:#6e7681">${e.t}</span> ${e.msg}</div>`).join('');
  rtChart(d.history);
}
setInterval(rtTick,1500); rtTick();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _params(self):
        d = {}
        if "?" in self.path:
            for kv in self.path.split("?", 1)[1].split("&"):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    d[k] = unquote(v)
        return d

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/videos":
            self._send(200, "application/json",
                       json.dumps(listar_videos()).encode("utf-8"))
            return
        if path == "/esp_data":
            with esp_lock:
                body = json.dumps(esp_state).encode("utf-8")
            self._send(200, "application/json", body)
            return
        if path == "/video_stream":
            p = self._params()
            try:
                self.send_response(200)
                self.send_header("Content-Type",
                                 "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                for chunk in gerar_stream_video(p.get("path", ""), p.get("modo", "rgb")):
                    self.wfile.write(chunk)
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                pass
            except Exception:
                pass
            return
        self._send(200, "text/html; charset=utf-8", PAGE.encode("utf-8"))

    def do_POST(self):
        if self.path.startswith("/esp_connect"):
            p = self._params()
            ok, msg = esp_start(p.get("port", "COM3"))
            self._send(200, "application/json",
                       json.dumps({"ok": ok, "msg": msg}).encode("utf-8"))
            return
        if not self.path.startswith("/detectar"):
            self._send(404, "text/plain", b"nao encontrado")
            return
        try:
            modo = "rgb"
            if "modo=" in self.path:
                modo = self.path.split("modo=")[1].split("&")[0]
            n = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(n)
            img = decode_image_bytes(raw)
            if img is None:
                raise ValueError("formato de imagem nao suportado")
            res = processar_imagem(img, modo)
            self._send(200, "application/json", json.dumps(res).encode("utf-8"))
        except Exception as e:
            self._send(200, "application/json",
                       json.dumps({"erro": str(e)}).encode("utf-8"))


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", WEB_PORT), Handler)
    print("=" * 56)
    print("  Frontend de Apresentacao — Detector de Placas")
    print(f"  Modelo : {os.path.basename(MODEL_PATH)} (LiteRT)")
    print(f"  Abra no navegador:  http://localhost:{WEB_PORT}")
    print("  (Ctrl+C para encerrar)")
    print("=" * 56)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nEncerrado.")


if __name__ == "__main__":
    main()
