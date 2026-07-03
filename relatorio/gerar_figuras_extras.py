#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera as figuras adicionais do relatorio (fundo claro, IEEE):

  figuras/fig_preproc.png          pre-processamento passo a passo (figure*)
  figuras/fig_seg_etapas.png       segmentacao com intermediarios + histograma Otsu
  figuras/fig_seg_exemplos.png     segmentacao nas 3 imagens de teste
  figuras/fig_tempo_log.png        tempos de inferencia por plataforma (log)
  figuras/fig_confianca_campo.png  serie real de confianca na ESP (02/07)

Complementa gerar_figuras.py (que extrai as figuras do video).
Uso:  python relatorio/gerar_figuras_extras.py
"""
import os
import sys

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
FIGDIR = os.path.join(HERE, "figuras")
sys.path.insert(0, os.path.join(REPO, "frontend"))
sys.path.insert(0, os.path.join(REPO, "apresentacao", "kit_slides"))
import frontend_apresentacao as fa            # noqa: E402
import gerar_assets as kit                    # noqa: E402  (fig_confianca_campo light)

# paleta light validada (dataviz): azul #2a78d6, vermelho #e34948, tinta #0b0b0b
AZUL, VERM = "#2a78d6", "#e34948"
INK, INK2, MUTED, GRIDC = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "text.color": INK, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "font.size": 9, "font.family": "sans-serif",
})

IMG_DEMO = os.path.join(REPO, "imagens_teste", "placa_demo_1_conf1.00.jpg")


def _off(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# ------------------------------------------------- 1) pre-processamento
def fig_preproc():
    bgr = cv2.imread(IMG_DEMO)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (224, 224))
    norm = (resized.astype(np.float32) / 127.5) - 1.0
    gray3 = cv2.cvtColor(cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY),
                         cv2.COLOR_GRAY2RGB)
    aug = cv2.convertScaleAbs(cv2.flip(resized, 1), alpha=1.15, beta=18)

    paineis = [
        (rgb, "(a) original"),
        (resized, "(b) 224×224"),
        (norm.mean(-1), "(c) normalizada $[-1,1]$"),
        (gray3, "(d) gray3 (luminância ×3)"),
        (aug, "(e) augmentada"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(7.16, 1.75), dpi=300)
    for ax, (im, tt) in zip(axes, paineis):
        if im.ndim == 2:
            ax.imshow(im, cmap="RdBu_r", vmin=-1, vmax=1)
        else:
            ax.imshow(im)
        ax.set_title(tt, fontsize=8)
        _off(ax)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(FIGDIR, "fig_preproc.png"), bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------- 2) segmentacao (etapas)
def fig_seg_etapas():
    bgr = cv2.imread(IMG_DEMO)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    _, bbox, conf, _, _ = fa.detectar(bgr)

    # mesmos passos/parametros de fa.segmentar_roi_placa, expostos
    h_img, w_img = rgb.shape[:2]
    x1, y1, x2, y2 = fa.bbox_yolo_para_pixel(bbox, w_img, h_img)
    x1, y1, x2, y2 = fa.expandir_bbox_pixel(x1, y1, x2, y2, w_img, h_img, 0.06)
    roi = rgb[y1:y2, x1:x2]
    h0, w0 = roi.shape[:2]
    roi_up = cv2.resize(roi, (w0 * 4, h0 * 4), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(roi_up, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4)).apply(gray)
    suave = cv2.bilateralFilter(clahe, 5, 50, 50)
    h, w = suave.shape[:2]
    kb = cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, w // 6), max(3, h // 4)))
    blackhat = cv2.morphologyEx(suave, cv2.MORPH_BLACKHAT, kb)
    otsu_val, m1 = cv2.threshold(blackhat, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, mask, seg = fa.segmentar_roi_placa(rgb, bbox)

    fig, axes = plt.subplots(2, 3, figsize=(3.5, 2.5), dpi=300)
    itens = [
        (roi_up, None, "(a) ROI da caixa"),
        (clahe, "gray", "(b) CLAHE"),
        (blackhat, "gray", "(c) BlackHat"),
        (None, None, "(d) histograma + Otsu"),
        (mask, "gray", "(e) máscara final"),
        (seg, None, "(f) segmentada"),
    ]
    for ax, (im, cm, tt) in zip(axes.ravel(), itens):
        if im is None:  # histograma do blackhat com limiar de Otsu
            ax.hist(blackhat.ravel(), bins=64, color=AZUL)
            ax.axvline(otsu_val, color=VERM, ls="--", lw=1.0)
            ax.text(otsu_val + 4, ax.get_ylim()[1] * 0.62,
                    f"Otsu = {otsu_val:.0f}", fontsize=6, color=VERM)
            ax.set_yscale("log")
            ax.tick_params(labelsize=5)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
        else:
            ax.imshow(im, cmap=cm)
            ax.set_xticks([]); ax.set_yticks([])
            for s_ in ax.spines.values():
                s_.set_visible(False)
        ax.set_title(tt, fontsize=6.5, pad=2)
    fig.tight_layout(pad=0.35)
    fig.savefig(os.path.join(FIGDIR, "fig_seg_etapas.png"), bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------- 3) segmentacao (exemplos)
def fig_seg_exemplos():
    fotos = sorted(os.listdir(os.path.join(REPO, "imagens_teste")))[:3]
    fig, axes = plt.subplots(3, 2, figsize=(3.5, 3.4), dpi=300)
    for lin, nome in enumerate(fotos):
        bgr = cv2.imread(os.path.join(REPO, "imagens_teste", nome))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pred, bbox, conf, _, vis = fa.detectar(bgr)
        box = vis.copy()
        x1, y1, x2, y2 = fa.bbox_yolo_para_pixel(bbox, fa.IMG_SIZE, fa.IMG_SIZE)
        cv2.rectangle(box, (x1, y1), (x2, y2), (42, 120, 214), 2)
        _, _, seg = fa.segmentar_roi_placa(rgb, bbox)
        axes[lin, 0].imshow(box)
        axes[lin, 0].set_title(f"detecção (conf = {conf:.2f})", fontsize=6.5, pad=2)
        axes[lin, 1].imshow(seg)
        axes[lin, 1].set_title("caracteres segmentados", fontsize=6.5, pad=2)
        for ax in axes[lin]:
            _off(ax)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(FIGDIR, "fig_seg_exemplos.png"), bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------- 4) tempos (light)
def fig_tempo_log():
    plats = ["Keras f32 (PC)", "TFLite f32 (PC)", "TFLite INT8 (PC)",
             "TFLite INT8 (ESP32-S3)"]
    ms = [96.9, 6.5, 5.5, 117_000.0]
    labels = ["96,9 ms", "6,5 ms", "4–7 ms", "117 s"]
    cores = [MUTED, MUTED, MUTED, AZUL]
    fig, ax = plt.subplots(figsize=(3.5, 1.9), dpi=300)
    y = np.arange(len(plats))[::-1]
    ax.barh(y, ms, height=0.55, color=cores, zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(1, 6e5)
    ax.set_yticks(y, plats, fontsize=7)
    ax.set_xticks([1, 100, 10_000, 1_000_000],
                  ["1 ms", "100 ms", "10 s", ""])
    ax.tick_params(labelsize=6.5)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.grid(True, axis="x", color=GRIDC, lw=0.6)
    ax.set_axisbelow(True)
    for yi, v, lb, c in zip(y, ms, labels, cores):
        ax.text(v * 1.35, yi, lb, va="center", fontsize=7,
                fontweight="bold", color=INK if c == AZUL else INK2)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(FIGDIR, "fig_tempo_log.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIGDIR, exist_ok=True)
    fig_preproc()
    fig_seg_etapas()
    fig_seg_exemplos()
    fig_tempo_log()
    kit.fig_confianca_campo(dark=False,
                            out=os.path.join(FIGDIR, "fig_confianca_campo.png"),
                            figsize=(7.16, 3.1), dpi=300, fs=0.62)
    print("Figuras extras geradas em", FIGDIR)
