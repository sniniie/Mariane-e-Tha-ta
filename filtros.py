
import cv2
import numpy as np
from PIL import Image


def pixel_art(imagem):
    img = np.array(imagem)
    h, w = img.shape[:2]

    pequena = cv2.resize(
        img,
        (max(1, w // 18), max(1, h // 18)),
        interpolation=cv2.INTER_LINEAR
    )

    resultado = cv2.resize(
        pequena,
        (w, h),
        interpolation=cv2.INTER_NEAREST
    )

    return Image.fromarray(resultado)


def anime(imagem):
    img = np.array(imagem)

    # Suavizacao das cores
    cores = cv2.bilateralFilter(img, 9, 200, 200)
    cores = cv2.bilateralFilter(cores, 9, 200, 200)

    # Criacao dos contornos
    cinza = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cinza = cv2.medianBlur(cinza, 5)

    bordas = cv2.adaptiveThreshold(
        cinza, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 9, 9
    )

    resultado = cv2.bitwise_and(
        cores, cores, mask=bordas
    )

    return Image.fromarray(resultado)


def cartoon(imagem):
    img = np.array(imagem)

    cores = cv2.bilateralFilter(img, 9, 250, 250)

    cinza = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cinza = cv2.medianBlur(cinza, 7)

    bordas = cv2.adaptiveThreshold(
        cinza, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 9, 9
    )

    resultado = cv2.bitwise_and(
        cores, cores, mask=bordas
    )

    return Image.fromarray(resultado)


def aquarela(imagem):
    img = np.array(imagem)

    resultado = cv2.stylization(
        img,
        sigma_s=100,
        sigma_r=0.45
    )

    return Image.fromarray(resultado)


def desenho(imagem):
    img = np.array(imagem)

    cinza = cv2.cvtColor(
        img, cv2.COLOR_RGB2GRAY
    )

    desenho, _ = cv2.pencilSketch(
        img,
        sigma_s=60,
        sigma_r=0.07,
        shade_factor=0.05
    )

    return Image.fromarray(desenho)


def espelhamento(imagem):
    img = np.array(imagem)

    h, w = img.shape[:2]

    metade = img[:, :w // 2]

    reflexo = cv2.flip(metade, 1)

    resultado = np.concatenate(
        (metade, reflexo), axis=1
    )

    return Image.fromarray(resultado)


def neon(imagem):
    img = np.array(imagem)

    cinza = cv2.cvtColor(
        img, cv2.COLOR_RGB2GRAY
    )

    bordas = cv2.Canny(cinza, 80, 160)

    brilho = cv2.GaussianBlur(
        bordas, (0, 0), 5
    )

    resultado = np.zeros_like(img)

    # Contornos coloridos
    resultado[:, :, 0] = brilho
    resultado[:, :, 1] = bordas
    resultado[:, :, 2] = bordas

    return Image.fromarray(resultado)


FILTROS = {
    "Pixel Art": pixel_art,
    "Anime": anime,
    "Cartoon": cartoon,
    "Aquarela": aquarela,
    "Desenho": desenho,
    "Espelhamento": espelhamento,
    "Neon": neon
}
