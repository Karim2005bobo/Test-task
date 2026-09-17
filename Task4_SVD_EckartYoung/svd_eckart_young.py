"""
Теорема Эккарта-Юнга (Eckart-Young): экспериментальная проверка.

Среди всех матриц ранга <= k усечённое SVD A_k = U_k Sigma_k V_k^T
минимизирует ||A - B||_F, и минимальная ошибка равна
    ||A - A_k||_F = sqrt(sum_{i>k} sigma_i^2)
— норме "хвоста" отброшенных сингулярных чисел.

Скрипт:
  1. Строит матрицу-"изображение" (scipy.datasets.face, с синтетическим
     фолбэком, если датасет недоступен).
  2. Считает SVD, rank-k приближения для набора k и сверяет реальную ошибку
     с теоретической формулой (с точностью до чисел с плавающей точкой).
  3. Для нескольких k берёт случайную факторизацию A ~ W1 @ W2 ранга k,
     дообучает её градиентным спуском (Adam) по ||A - W1 W2||_F и
     показывает, что оптимизированная ошибка не может превзойти ошибку
     SVD-усечения того же ранга (и приближается к ней снизу).
  4. Строит графики: ошибка(k) с теоретической кривой и compression ratio(k).

Все результаты (числа, графики) сохраняются в figures/ и results.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)

RNG_SEED = 0
np.random.seed(RNG_SEED)


# ----------------------------------------------------------------------
# 1. Матрица-изображение
# ----------------------------------------------------------------------
def load_image() -> np.ndarray:
    """Загружает raccoon-face из scipy.datasets и уменьшает его для скорости
    последующего градиентного спуска. При отсутствии сети/пакета pooch
    строит детерминированный синтетический пример."""
    try:
        from scipy.datasets import face

        img = face(gray=True).astype(np.float64)
        # block-average downsample 768x1024 -> 96x128, чтобы GD-часть была быстрой
        factor = 8
        h, w = img.shape
        img = img[: h - h % factor, : w - w % factor]
        img = img.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))
        source = "scipy.datasets.face(gray=True), downsampled 8x"
    except Exception as exc:  # noqa: BLE001
        print(f"[!] scipy.datasets.face недоступен ({exc}); строю синтетический пример")
        h, w = 96, 128
        y, x = np.mgrid[0:h, 0:w]
        img = (
            120
            + 60 * np.sin(x / 6.0)
            + 40 * np.cos(y / 9.0)
            + 30 * np.exp(-((x - w / 2) ** 2 + (y - h / 2) ** 2) / (2 * (w / 5) ** 2))
        )
        source = "synthetic deterministic pattern"
    return img, source


A, source_desc = load_image()
m, n = A.shape
r_full = min(m, n)
print(f"Матрица A: {A.shape}, источник: {source_desc}")


# ----------------------------------------------------------------------
# 2. SVD, rank-k приближения, сверка с теоретической формулой
# ----------------------------------------------------------------------
U, s, Vt = np.linalg.svd(A, full_matrices=False)  # s: убывающие сингулярные числа

def truncated_svd_approx(k: int) -> np.ndarray:
    return (U[:, :k] * s[:k]) @ Vt[:k, :]


def theoretical_tail_error(k: int) -> float:
    return np.sqrt(np.sum(s[k:] ** 2))


ks = sorted(set([0, 1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 60, 80, r_full // 2, r_full]))
ks = [k for k in ks if 0 <= k <= r_full]

rows = []
for k in ks:
    Ak = truncated_svd_approx(k)
    real_err = np.linalg.norm(A - Ak, ord="fro")
    theo_err = theoretical_tail_error(k)
    rows.append((k, real_err, theo_err, abs(real_err - theo_err)))

print("\nk   ||A-A_k||_F (реальная)   sqrt(sum tail sigma^2) (теория)   |разница|")
for k, real_err, theo_err, diff in rows:
    print(f"{k:3d}   {real_err:22.10f}   {theo_err:28.10f}   {diff:.3e}")

max_diff = max(r[3] for r in rows)
assert max_diff < 1e-8 * (s[0] + 1), f"Слишком большое расхождение: {max_diff}"
print(f"\nМаксимальное расхождение реальной ошибки и теоретической формулы: {max_diff:.3e}"
      " (совпадает с точностью до чисел с плавающей точкой)")


# ----------------------------------------------------------------------
# 3. Проверка оптимальности: случайная факторизация + градиентный спуск (Adam)
# ----------------------------------------------------------------------
def frob_error(B: np.ndarray) -> float:
    return np.linalg.norm(A - B, ord="fro")


def train_factorization(k: int, n_iters: int = 3000, lr: float = 1.0, seed: int = 1):
    """Учит W1 (m x k), W2 (k x n) минимизировать ||A - W1 W2||_F^2 через Adam,
    инициализация случайная (не имеет отношения к SVD).

    Масштаб инициализации подобран так, чтобы std(W1 @ W2) ~ std(A): для
    W_{il}, W_{jl} ~ N(0, sigma^2) имеем std((W1 W2)_{ij}) = sigma^2 * sqrt(k),
    откуда sigma = sqrt(std(A) / sqrt(k)). Без этого случайная инициализация
    даёт продукт на порядки больше/меньше A, и Adam сходится слишком медленно
    за разумное число итераций."""
    rng = np.random.default_rng(seed)
    sigma = np.sqrt(np.std(A) / np.sqrt(k))
    W1 = rng.normal(scale=sigma, size=(m, k))
    W2 = rng.normal(scale=sigma, size=(k, n))

    # Adam state
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    mW1, vW1 = np.zeros_like(W1), np.zeros_like(W1)
    mW2, vW2 = np.zeros_like(W2), np.zeros_like(W2)

    loss_history = []
    for t in range(1, n_iters + 1):
        R = A - W1 @ W2  # residual
        loss = np.sum(R ** 2)
        loss_history.append(np.sqrt(loss))

        gW1 = -2.0 * R @ W2.T
        gW2 = -2.0 * W1.T @ R

        mW1 = beta1 * mW1 + (1 - beta1) * gW1
        vW1 = beta2 * vW1 + (1 - beta2) * gW1 ** 2
        mW2 = beta1 * mW2 + (1 - beta1) * gW2
        vW2 = beta2 * vW2 + (1 - beta2) * gW2 ** 2

        mW1_hat = mW1 / (1 - beta1 ** t)
        vW1_hat = vW1 / (1 - beta2 ** t)
        mW2_hat = mW2 / (1 - beta1 ** t)
        vW2_hat = vW2 / (1 - beta2 ** t)

        W1 -= lr * mW1_hat / (np.sqrt(vW1_hat) + eps)
        W2 -= lr * mW2_hat / (np.sqrt(vW2_hat) + eps)

    final_err = frob_error(W1 @ W2)
    return W1, W2, np.array(loss_history), final_err


gd_ks = [2, 5, 10, 20, 40]
gd_results = {}
print("\nГрадиентный спуск (Adam) vs SVD-усечение того же ранга:")
print("k     SVD error         GD error (после обучения)   GD - SVD (>= 0 ожидается)")
for k in gd_ks:
    W1, W2, hist, gd_err = train_factorization(k)
    svd_err = theoretical_tail_error(k)
    gap = gd_err - svd_err
    gd_results[k] = dict(hist=hist, gd_err=gd_err, svd_err=svd_err, gap=gap)
    print(f"{k:3d}   {svd_err:14.6f}   {gd_err:22.6f}   {gap:+.6e}")
    # небольшой допуск на численный шум оптимизации/floating-point, не на "победу" над SVD
    assert gap > -1e-3 * max(svd_err, 1.0), (
        f"GD побил SVD при k={k}! Это противоречит теореме Эккарта-Юнга."
    )

print(
    "\nВо всех случаях градиентный спуск НЕ смог превзойти ошибку SVD-усечения "
    "(gap >= 0 с точностью до погрешности оптимизации), что подтверждает "
    "оптимальность усечённого SVD экспериментально."
)


# ----------------------------------------------------------------------
# 4. Графики
# ----------------------------------------------------------------------
plt.rcParams.update({"figure.dpi": 110, "font.size": 10})

# 4a. Оригинал + несколько rank-k приближений
show_ks = [1, 3, 5, 10, 20, r_full]
fig, axes = plt.subplots(1, len(show_ks) + 1, figsize=(3 * (len(show_ks) + 1), 3.3))
axes[0].imshow(A, cmap="gray")
axes[0].set_title("Оригинал")
axes[0].axis("off")
for ax, k in zip(axes[1:], show_ks):
    Ak = truncated_svd_approx(k)
    ax.imshow(Ak, cmap="gray")
    err = frob_error(Ak)
    ax.set_title(f"k={k}\n||A-A_k||_F={err:.1f}")
    ax.axis("off")
fig.suptitle("Усечённое SVD-приближение изображения при разных рангах k")
fig.tight_layout()
fig.savefig(FIG_DIR / "01_rank_k_approximations.png")
plt.close(fig)

# 4b. Ошибка(k): реальная vs теоретическая + compression ratio(k)
all_ks = np.arange(0, r_full + 1)
real_errs = np.array([frob_error(truncated_svd_approx(k)) for k in all_ks])
theo_errs = np.array([theoretical_tail_error(k) for k in all_ks])
storage_full = m * n
storage_k = all_ks * (m + n + 1)  # U_k (m x k) + V_k (n x k) + sigma_k (k)
compression_ratio = storage_k / storage_full  # доля от исходного объёма памяти

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

ax1.plot(all_ks, theo_errs, color="tab:blue", lw=2.5, label="теория: sqrt(sum_{i>k} sigma_i^2)")
ax1.scatter([r[0] for r in rows], [r[1] for r in rows], color="tab:red", zorder=5,
            marker="x", s=60, label="реальная ||A-A_k||_F (выборочные k)")
for k in gd_ks:
    ax1.scatter(k, gd_results[k]["gd_err"], color="tab:green", marker="o", zorder=6)
ax1.scatter([], [], color="tab:green", marker="o", label="ошибка после GD-факторизации")
ax1.set_xlabel("ранг k")
ax1.set_ylabel("ошибка Фробениуса")
ax1.set_yscale("log")
ax1.set_title("Ошибка приближения от ранга k")
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

ax2.plot(all_ks, compression_ratio, color="tab:purple", lw=2.5)
ax2.axhline(1.0, color="gray", ls="--", lw=1)
ax2.set_xlabel("ранг k")
ax2.set_ylabel("compression ratio  =  k(m+n+1) / (m*n)")
ax2.set_title("Отношение объёма хранения rank-k факторизации\nк объёму исходной матрицы")
ax2.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(FIG_DIR / "02_error_and_compression_vs_k.png")
plt.close(fig)

# 4c. Кривые обучения GD и сравнение с SVD-порогом для каждого k
fig, axes = plt.subplots(1, len(gd_ks), figsize=(3.2 * len(gd_ks), 3.2), sharey=False)
for ax, k in zip(axes, gd_ks):
    res = gd_results[k]
    ax.plot(res["hist"], color="tab:green", label="||A - W1 W2||_F (GD)")
    ax.axhline(res["svd_err"], color="tab:blue", ls="--", label="SVD-порог (оптимум)")
    ax.set_title(f"k={k}")
    ax.set_xlabel("итерация")
    if ax is axes[0]:
        ax.set_ylabel("ошибка Фробениуса")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
fig.suptitle("Сходимость градиентного спуска (Adam) к SVD-оптимуму того же ранга")
fig.tight_layout()
fig.savefig(FIG_DIR / "03_gd_convergence_vs_svd.png")
plt.close(fig)

print(f"\nГрафики сохранены в {FIG_DIR}")


# ----------------------------------------------------------------------
# Сохранение числовых результатов
# ----------------------------------------------------------------------
results = {
    "matrix_shape": [m, n],
    "source": source_desc,
    "singular_values_head": s[:10].tolist(),
    "eckart_young_check": [
        {"k": k, "real_error": real_err, "theoretical_error": theo_err, "abs_diff": diff}
        for k, real_err, theo_err, diff in rows
    ],
    "max_abs_diff_svd_vs_theory": max_diff,
    "gd_vs_svd": [
        {
            "k": k,
            "svd_error": gd_results[k]["svd_err"],
            "gd_error": gd_results[k]["gd_err"],
            "gap_gd_minus_svd": gd_results[k]["gap"],
        }
        for k in gd_ks
    ],
}
with open(HERE / "results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"Числовые результаты сохранены в {HERE / 'results.json'}")
