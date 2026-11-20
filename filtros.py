import os
import urllib.request
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

MODELO_PATH = "AnimeGANv2_Hayao.onnx"
MODELO_URL = "https://github.com/TachibanaYoshino/AnimeGANv2/releases/download/1.0/AnimeGANv2_Hayao.onnx"

_sessao_anime = None

def _carregar_modelo_anime():
    global _sessao_anime
    if _sessao_anime is None:
        if not os.path.exists(MODELO_PATH):
            urllib.request.urlretrieve(MODELO_URL, MODELO_PATH)
        _sessao_anime = ort.InferenceSession(MODELO_PATH, providers=["CPUExecutionProvider"])
    return _sessao_anime

def _converter_para_bgr(imagem):
    img = np.array(imagem)
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def _retornar_rgb(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

def anime(imagem):
    img_rgb = np.array(imagem.convert("RGB"))
    h, w, _ = img_rgb.shape

    novo_w = max(256, (w // 32) * 32)
    novo_h = max(256, (h // 32) * 32)
    img_redim = cv2.resize(img_rgb, (novo_w, novo_h))

    img_input = img_redim.astype(np.float32) / 127.5 - 1.0
    img_input = np.expand_dims(img_input, axis=0)

    sessao = _carregar_modelo_anime()
    input_name = sessao.get_inputs()[0].name
    saida = sessao.run(None, {input_name: img_input})[0][0]

    saida = (saida + 1.0) * 127.5
    saida = np.clip(saida, 0, 255).astype(np.uint8)
    saida = cv2.resize(saida, (w, h))
    return Image.fromarray(saida)

def pixel_art(imagem):
    img_bgr = _converter_para_bgr(imagem)
    h, w = img_bgr.shape[:2]
    fator = max(1, min(w, h) // 30)
    pequena = cv2.resize(img_bgr, (max(1, w // fator), max(1, h // fator)), interpolation=cv2.INTER_LINEAR)
    resultado = cv2.resize(pequena, (w, h), interpolation=cv2.INTER_NEAREST)
    return _retornar_rgb(resultado)

def cartoon(imagem):
    img_bgr = _converter_para_bgr(imagem)
    cinza = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cinza = cv2.medianBlur(cinza, 5)
    bordas = cv2.adaptiveThreshold(cinza, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    colorido = cv2.bilateralFilter(img_bgr, 9, 300, 300)
    resultado = cv2.bitwise_and(colorido, colorido, mask=bordas)
    return _retornar_rgb(resultado)

def aquarela(imagem):
    img_bgr = _converter_para_bgr(imagem)
    resultado = cv2.stylization(img_bgr, sigma_s=60, sigma_r=0.6)
    return _retornar_rgb(resultado)

def desenho(imagem):
    img_bgr = _converter_para_bgr(imagem)
    _, sketch = cv2.pencilSketch(img_bgr, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
    resultado = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    return _retornar_rgb(resultado)

def espelhamento(imagem):
    img_bgr = _converter_para_bgr(imagem)
    resultado = cv2.flip(img_bgr, 1)
    return _retornar_rgb(resultado)

def neon(imagem):
    img_bgr = _converter_para_bgr(imagem)
    cinza = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(cinza, (21, 21), 0)
    invertido = 255 - blur
    sketch = cv2.divide(cinza, 255 - invertido, scale=256)
    resultado = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    resultado = cv2.applyColorMap(resultado, cv2.COLORMAP_JET)
    return _retornar_rgb(resultado)
