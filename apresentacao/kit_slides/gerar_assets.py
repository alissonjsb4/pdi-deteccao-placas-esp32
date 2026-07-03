#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera os assets (PNG 16:9, fundo escuro) do kit de slides da apresentacao.

Reusa o pipeline real do frontend (LiteRT + OpenCV) — nada e mockado:
  - aba_imagem_*.png : deteccao + heatmap + segmentacao nas imagens de teste
  - aba_video.png    : melhor frame anotado do video de teste
  - fig_tempo_log.png: tempos de inferencia por plataforma (escala log)
  - fig_confianca_campo.png: serie real de confianca medida na ESP em 02/07/2026
  - fig_pipeline_firmware.png: diagrama de blocos do loop() do firmware

Uso:  python apresentacao/kit_slides/gerar_assets.py
"""
import os
import sys
from datetime import datetime

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "frontend"))
import frontend_apresentacao as fa  # noqa: E402  (carrega o modelo LiteRT)

# ---------------------------------------------------------------- paleta
# Validada com scripts/validate_palette.js sobre a superficie navy do deck:
# accent #14a87c (banda L ok, >=3:1), referencia #e66767 (>=3:1);
# #8fa0b5 e o cinza de de-enfase (forma "emphasis"), nao um slot categorico.
NAVY = "#0e2036"      # superficie (casa com o deck Canva)
MINT = "#14a87c"      # accent (familia do teal do deck)
GRAY = "#8fa0b5"      # de-enfase / muted ink
INK = "#ffffff"      # primario
INK2 = "#c3ccd9"      # secundario
GRID = "#24344d"      # gridline recessiva
RED = "#e66767"      # linha de referencia (limiar)

plt.rcParams.update({
    "figure.facecolor": NAVY, "axes.facecolor": NAVY, "savefig.facecolor": NAVY,
    "text.color": INK2, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": GRAY, "ytick.color": GRAY, "font.size": 14,
    "font.family": "sans-serif",
})

FIGSIZE = (12.8, 7.2)  # 16:9
DPI = 150


def _ax_clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- 1) tempos
def fig_tempo_log():
    plats = ["Keras float32 (PC)", "TFLite float32 (PC)",
             "TFLite INT8 (PC, LiteRT)", "TFLite INT8 (ESP32-S3)"]
    ms = [96.9, 6.5, 5.5, 117_000.0]
    labels = ["96,9 ms", "6,5 ms", "4–7 ms", "117 s"]
    cores = [GRAY, GRAY, GRAY, MINT]

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    y = np.arange(len(plats))[::-1]
    ax.barh(y, ms, height=0.52, color=cores, zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(1, 4e5)
    ax.set_yticks(y, plats, fontsize=15)
    ax.set_xticks([1, 10, 100, 1000, 10_000, 100_000],
                  ["1 ms", "10 ms", "100 ms", "1 s", "10 s", "100 s"])
    _ax_clean(ax)
    ax.grid(False, axis="y")
    for yi, v, lb, c in zip(y, ms, labels, cores):
        ax.text(v * 1.25, yi, lb, va="center", fontsize=16, fontweight="bold",
                color=INK if c == MINT else INK2)
    ax.annotate("mesmo arquivo .tflite (2,35 MB):\n~17.000× mais lento no microcontrolador",
                xy=(117_000, y[3]), xytext=(3000, y[3] + 0.85),
                fontsize=13, color=INK2,
                arrowprops=dict(arrowstyle="-", color=GRAY, lw=1))
    fig.text(0.04, 0.955, "Tempo de inferência — mesmo modelo, plataformas diferentes",
             color=INK, fontsize=20, fontweight="bold")
    fig.text(0.04, 0.905, "escala logarítmica; medições reais do trabalho",
             fontsize=12, color=GRAY)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    fig.savefig(os.path.join(HERE, "fig_tempo_log.png"))
    plt.close(fig)


# ---------------------------------------------------------------- 2) campo
# Serie real medida em 02/07/2026 (dashboard /esp_data do frontend).
CAMPO = [  # (hora, confianca)
    ("20:07:08", 0.86), ("20:09:05", 0.89), ("20:11:02", 0.87),
    ("20:15:16", 0.82), ("20:17:14", 0.92), ("20:19:11", 0.88),
    ("20:23:26", 0.88), ("20:25:23", 0.83), ("20:27:20", 0.84),
    ("20:29:18", 0.84), ("20:31:15", 0.88), ("20:33:12", 0.90),
]
REGRAVACOES = [("20:13:19", "0,90"), ("20:21:29", "0,85")]  # recalibracao do limiar
TELEGRAM = "20:33:30"


def _t(hhmmss):
    return datetime(2026, 7, 2, *map(int, hhmmss.split(":")))


def fig_confianca_campo(dark=True, out=None, figsize=None, dpi=None, fs=1.0):
    """fs = fator de escala de fonte (menor para a figura do relatorio)."""
    if dark:
        surf, accent, ink, ink2, gray, grid, red = NAVY, MINT, INK, INK2, GRAY, GRID, RED
    else:  # versao clara p/ relatorio IEEE (paleta light validada)
        surf, accent, ink, ink2, gray, grid, red = ("#ffffff", "#2a78d6", "#0b0b0b",
                                                    "#52514e", "#898781", "#e1e0d9",
                                                    "#e34948")
    times = [_t(t) for t, _ in CAMPO]
    confs = [c for _, c in CAMPO]

    # limiar vigente em cada instante (recalibrado em campo)
    thr_t = [_t("20:05:00"), _t("20:13:19"), _t("20:21:29"), _t("20:35:30")]
    thr_v = [0.95, 0.90, 0.85, 0.85]

    def thr_at(t):
        v = thr_v[0]
        for ti, vi in zip(thr_t, thr_v):
            if t >= ti:
                v = vi
        return v

    fig, ax = plt.subplots(figsize=figsize or FIGSIZE, dpi=dpi or DPI)
    fig.patch.set_facecolor(surf); ax.set_facecolor(surf)

    # faixa "sem placa" medida (0,42-0,58) — contexto de separacao
    ax.axhspan(0.42, 0.58, color=gray, alpha=0.16, zorder=1)
    ax.text(times[0], 0.50, "faixa sem placa (medida): 0,42–0,58",
            fontsize=11 * fs, color=gray, va="center")

    # limiar (degraus) + linha da serie
    ax.step(thr_t, thr_v, where="post", color=red, ls="--", lw=1.6, zorder=2,
            label="limiar de disparo (recalibrado em campo)")
    ax.plot(times, confs, color=gray, lw=1.4, zorder=3)

    acima = [(t, c) for t, c in zip(times, confs) if c >= thr_at(t)]
    abaixo = [(t, c) for t, c in zip(times, confs) if c < thr_at(t)]
    if abaixo:
        ax.scatter(*zip(*abaixo), s=64 * fs, facecolors=surf, edgecolors=gray,
                   lw=1.6, zorder=4, label="inferência < limiar (rejeitada)")
    if acima:
        ax.scatter(*zip(*acima), s=80 * fs, color=accent, zorder=5,
                   label="inferência ≥ limiar")

    # regravacoes do firmware
    for t, novo in REGRAVACOES:
        ax.axvline(_t(t), color=grid, lw=1.2, ls=":")
        ax.text(_t(t), 0.985, f"regrava\nlimiar {novo}", fontsize=10 * fs,
                color=gray, ha="center", va="top")

    # disparo + Telegram
    ax.axvline(_t(TELEGRAM), color=accent, lw=1.4, ls=":")
    ax.annotate("2 frames seguidos ≥ limiar\n→ foto no Telegram (+18 s)",
                xy=(_t(TELEGRAM), 0.90), xytext=(_t("20:26:40"), 0.965),
                fontsize=12 * fs, color=ink,
                arrowprops=dict(arrowstyle="->", color=accent, lw=1.2))

    ax.set_ylim(0.40, 1.0)
    ax.set_xlim(_t("20:05:00"), _t("20:35:50"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_ylabel("confiança da célula mais ativa", fontsize=12 * fs, color=ink2)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.grid(True, axis="y", color=grid, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=gray, labelsize=11 * fs)
    ax.set_title("Operação em campo (02/07) — uma inferência a cada ~117 s",
                 color=ink, fontsize=19 * fs, fontweight="bold", loc="left", pad=14)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=3,
              fontsize=11 * fs, frameon=False, labelcolor=ink2)
    fig.tight_layout()
    fig.savefig(out or os.path.join(HERE, "fig_confianca_campo.png"),
                facecolor=surf)
    plt.close(fig)


# ---------------------------------------------------------------- 3) pipeline
def fig_pipeline_firmware():
    passos_l1 = [
        ("Câmera OV5640\nJPEG 320×240", GRAY),
        ("fmt2rgb888\nJPEG → RGB", GRAY),
        ("resize 224×224\n+ quant. INT8", GRAY),
        ("Invoke()\nTFLite Micro\n~117 s", MINT),
    ]
    passos_l2 = [
        ("decodifica grade\n7×7×5 → bbox", GRAY),
        ("filtro: conf ≥ limiar\npor 2 frames + cooldown", GRAY),
        ("desenha bbox\n+ fmt2jpg", GRAY),
        ("POST multipart\nTelegram (~18 s)", MINT),
    ]
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")

    def caixa(x, y, txt, cor):
        destaque = cor == MINT
        fb = FancyBboxPatch((x, y), 2.0, 1.35,
                            boxstyle="round,pad=0.10,rounding_size=0.12",
                            fc=(MINT if destaque else "#16283f"),
                            ec=(MINT if destaque else GRID), lw=1.6, zorder=3)
        ax.add_patch(fb)
        ax.text(x + 1.0, y + 0.675, txt, ha="center", va="center",
                fontsize=13.5, fontweight="bold",
                color=(NAVY if destaque else INK2), zorder=4)

    def seta(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                                     arrowstyle="-|>", mutation_scale=22,
                                     color=GRAY, lw=1.6, zorder=2))

    xs = [0.4, 2.9, 5.4, 7.9]
    for x, (txt, cor) in zip(xs, passos_l1):
        caixa(x, 3.55, txt, cor)
    for a, b in zip(xs, xs[1:]):
        seta(a + 2.12, 4.23, b - 0.12, 4.23)
    # curva da linha 1 para a linha 2 (Invoke -> decodifica)
    ax.plot([8.9, 8.9], [3.42, 2.95], color=GRAY, lw=1.6, zorder=2)
    ax.plot([8.9, 1.4], [2.95, 2.95], color=GRAY, lw=1.6, zorder=2)
    seta(1.4, 2.95, 1.4, 2.48)
    xs2 = [0.4, 2.9, 5.4, 7.9]
    for x, (txt, cor) in zip(xs2, passos_l2):
        caixa(x, 0.9, txt, cor)
    for a, b in zip(xs2, xs2[1:]):
        seta(a + 2.12, 1.58, b - 0.12, 1.58)

    ax.text(0.4, 5.68, "Firmware na ESP32-S3 — loop() do Identificador_de_placas.ino",
            fontsize=20, fontweight="bold", color=INK)
    ax.text(0.4, 5.30, "arena de 3 MB na PSRAM · partição custom de 5 MB · "
                       "modelo INT8 de 2,35 MB embutido como header C++",
            fontsize=12.5, color=GRAY)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_pipeline_firmware.png"))
    plt.close(fig)


# ---------------------------------------------------------------- 4) abas
def _card(imgs_rgb, titulos, out, titulo, altura=430):
    """Compoe painel horizontal estilo card escuro 16:9."""
    n = len(imgs_rgb)
    fig, axes = plt.subplots(1, n, figsize=FIGSIZE, dpi=DPI)
    if n == 1:
        axes = [axes]
    for ax, im, tt in zip(axes, imgs_rgb, titulos):
        ax.imshow(im)
        ax.set_title(tt, fontsize=15, color=INK2, pad=10)
        ax.axis("off")
    fig.suptitle(titulo, fontsize=20, fontweight="bold", color=INK, x=0.02,
                 ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out)
    plt.close(fig)


def assets_aba_imagem():
    for i, nome in enumerate(sorted(os.listdir(os.path.join(REPO, "imagens_teste"))), 1):
        p = os.path.join(REPO, "imagens_teste", nome)
        img = cv2.imread(p)
        if img is None:
            continue
        pred, bbox, conf, cell, vis = fa.detectar(img)
        box = vis.copy()
        x1, y1, x2, y2 = fa.bbox_yolo_para_pixel(bbox, fa.IMG_SIZE, fa.IMG_SIZE)
        cv2.rectangle(box, (x1, y1), (x2, y2), (0, 220, 120), 2)
        heat = fa.heatmap_rgb(pred)
        rgb_full = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        roi, mask, seg = fa.segmentar_roi_placa(rgb_full, bbox)
        _card([box, heat, seg],
              [f"detecção (conf = {conf:.2f})", "mapa de confiança 7×7",
               "caracteres segmentados"],
              os.path.join(HERE, f"aba_imagem_{i}.png"),
              f"Aba Imagem — {nome}")


def asset_aba_video():
    vid = os.path.join(REPO, "videos_teste", "carro_garagem.mp4")
    cap = cv2.VideoCapture(vid)
    melhor, melhor_conf = None, -1.0
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % 5 == 0:
            _, _, conf, _, _ = fa.detectar(frame)
            if conf > melhor_conf:
                melhor_conf, melhor = conf, frame.copy()
        i += 1
    cap.release()
    out, conf, placa = fa.anotar_frame(melhor, "rgb", "")
    _card([cv2.cvtColor(out, cv2.COLOR_BGR2RGB)],
          [f"melhor frame do vídeo (conf = {conf:.3f})"],
          os.path.join(HERE, "aba_video.png"),
          "Aba Vídeo — detecção quadro a quadro (stream MJPEG)")


if __name__ == "__main__":
    fig_tempo_log()
    fig_confianca_campo()
    fig_pipeline_firmware()
    assets_aba_imagem()
    asset_aba_video()
    print("Assets gerados em", HERE)
