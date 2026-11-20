import cv2
import numpy as np
from PIL import Image

def _converter_para_bgr(imagem):
    img = np.array(imagem)
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def _retornar_rgb(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

def anime(imagem):
    img_bgr = _converter_para_bgr(imagem)
    h, w = img_bgr.shape[:2]
    
    # Efeito estilizado de anime usando processamento de imagem avançado do OpenCV
    num_down = 2
    num_bilateral = 7
    
    img_small = img_bgr
    for _ in range(num_down):
        img_small = cv2.pyrDown(img_small)
        
    for _ in range(num_bilateral):
        img_small = cv2.bilateralFilter(img_small, d=9, sigmaColor=9, sigmaSpace=7)
        
    for _ in range(num_down):
        img_small = cv2.pyrUp(img_small)
        
    img_small = cv2.resize(img_small, (w, h))
    
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    resultado = cv2.bitwise_and(img_small, edges)
    return _retornar_rgb(resultado)

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

FILTROS = {
    "Anime": anime,
    "Pixel Art": pixel_art,
    "Cartoon": cartoon,
    "Aquarela": aquarela,
    "Desenho": desenho,
    "Espelhamento": espelhamento,
    "Neon": neon
}
