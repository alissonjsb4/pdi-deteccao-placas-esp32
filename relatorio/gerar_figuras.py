#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera automaticamente as figuras do relatorio a partir do video de teste,
usando o MESMO modelo embarcado (via o pipeline de frontend_apresentacao.py).

- Varre todos os frames do video, roda a deteccao e escolhe:
    * o melhor frame (maior confianca) -> figuras de deteccao + segmentacao
    * um frame de baixa confianca       -> figura "sem placa" (rejeicao)
- Salva PNGs prontos em relatorio/figuras/.

Uso: python relatorio/gerar_figuras.py
"""
import os
import sys
import glob
import numpy as np
import cv2

# importa o pipeline ja pronto (deteccao + segmentacao + render)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "frontend"))
import frontend_apresentacao as F  # noqa: E402

OUT = os.path.join(HERE, "figuras")
os.makedirs(OUT, exist_ok=True)


def achar_video():
    for e in ("*.mp4", "*.mov", "*.avi", "*.mkv"):
        v = glob.glob(os.path.join(ROOT, "**", e), recursive=True)
        if v:
            return v[0]
    return None


def titulo(img_rgb, txt, alt=28):
    h, w = img_rgb.shape[:2]
    bar = np.full((alt, w, 3), 22, np.uint8)
    cv2.putText(bar, txt, (8, alt - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (230, 237, 243), 1, cv2.LINE_AA)
    return np.vstack([bar, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)])


def salvar(nome, img_bgr):
    p = os.path.join(OUT, nome)
    cv2.imwrite(p, img_bgr)
    print("  salvo:", os.path.relpath(p, ROOT))


def montar_painel(cells, cols):
    """cells: lista de imagens BGR ja com titulo (mesma largura)."""
    # normaliza largura
    w = max(c.shape[1] for c in cells)
    norm = []
    for c in cells:
        if c.shape[1] != w:
            c = cv2.copyMakeBorder(c, 0, 0, 0, w - c.shape[1],
                                   cv2.BORDER_CONSTANT, value=(13, 17, 23))
        norm.append(c)
    rows = []
    for i in range(0, len(norm), cols):
        grp = norm[i:i + cols]
        h = max(c.shape[0] for c in grp)
        grp = [cv2.copyMakeBorder(c, 0, h - c.shape[0], 0, 0,
               cv2.BORDER_CONSTANT, value=(13, 17, 23)) for c in grp]
        rows.append(np.hstack(grp))
    w2 = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, w2 - r.shape[1],
            cv2.BORDER_CONSTANT, value=(13, 17, 23)) for r in rows]
    return np.vstack(rows)


def main():
    vid = achar_video()
    if not vid:
        print("Nenhum video encontrado. Coloque um .mp4 no projeto.")
        return
    print("Video:", os.path.relpath(vid, ROOT))
    cap = cv2.VideoCapture(vid)
    melhor = {"conf": -1}
    pior = {"conf": 2}
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        pred, bbox, conf, cell, vis = F.detectar(frame, "rgb")
        if conf > melhor["conf"]:
            melhor = {"conf": conf, "frame": frame.copy(), "pred": pred,
                      "bbox": bbox, "cell": cell, "vis": vis, "idx": idx}
        if conf < pior["conf"]:
            pior = {"conf": conf, "frame": frame.copy(), "vis": vis, "idx": idx}
        idx += 1
    cap.release()
    print(f"Frames analisados: {idx}")
    print(f"Melhor frame: #{melhor['idx']} conf={melhor['conf']:.3f}")
    print(f"Pior frame:   #{pior['idx']} conf={pior['conf']:.3f}")

    # ---- figuras do MELHOR frame (placa) ----
    fr = melhor["frame"]; bbox = melhor["bbox"]; pred = melhor["pred"]
    vis = melhor["vis"]; conf = melhor["conf"]
    h, w = fr.shape[:2]
    cor = (0, 200, 0) if conf >= 0.95 else (255, 140, 0)

    # frame completo anotado (modo video / tempo real)
    full = fr.copy()
    x1, y1, x2, y2 = F.bbox_yolo_para_pixel(bbox, w, h)
    cv2.rectangle(full, (x1, y1), (x2, y2), cor[::-1], 3)
    cv2.rectangle(full, (0, 0), (w, 30), (0, 0, 0), -1)
    cv2.putText(full, f"conf={conf:.2f}  {'PLACA' if conf>=0.95 else 'sem placa'}",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor[::-1], 2)
    salvar("fig_frame_anotado.png", full)

    # painel de deteccao: entrada | bbox(224) | heatmap
    vbox = vis.copy()
    bx1, by1, bx2, by2 = F.bbox_yolo_para_pixel(bbox, F.IMG_SIZE, F.IMG_SIZE)
    cv2.rectangle(vbox, (bx1, by1), (bx2, by2), cor, 2)
    heat = F.heatmap_rgb(pred)
    det = montar_painel([
        titulo(vis, "Entrada 224x224"),
        titulo(vbox, f"Deteccao (conf={conf:.2f})"),
        titulo(heat, "Mapa de confianca (grade 7x7)"),
    ], cols=3)
    salvar("fig_deteccao.png", det)

    # painel de segmentacao: ROI | mascara | caracteres
    img_rgb_full = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
    roi, mask, seg = F.segmentar_roi_placa(img_rgb_full, bbox)
    mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    segp = montar_painel([
        titulo(roi, "ROI da placa"),
        titulo(mask3, "Mascara (segmentacao)"),
        titulo(seg, "Caracteres segmentados"),
    ], cols=3)
    salvar("fig_segmentacao.png", segp)

    # ---- figura do PIOR frame (sem placa / rejeicao) ----
    pf = pior["frame"]; pv = pior["vis"]; pc = pior["conf"]
    pann = pf.copy()
    cv2.rectangle(pann, (0, 0), (pf.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(pann, f"conf={pc:.2f}  abaixo de 0.95 (nao dispara)",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 140, 0), 2)
    salvar("fig_sem_placa.png", pann)

    print("\nFiguras geradas em relatorio/figuras/.")


if __name__ == "__main__":
    main()
