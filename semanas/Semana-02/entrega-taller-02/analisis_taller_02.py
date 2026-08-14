"""Análisis reproducible del Taller 2: eigenfaces sobre LFW.

Sigue los parámetros fijos de la guía del curso y evita fuga de información:
la media, la SVD y el clasificador se aprenden únicamente con entrenamiento.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_lfw_people
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.extmath import randomized_svd


SEED = 10101
K_GRID = [10, 25, 50, 100, 250, 500]
PILOT_USERS = ["George W Bush", "Serena Williams"]
ROOT = Path(__file__).resolve().parent
WEEK_DIR = ROOT.parent
DATA_HOME = WEEK_DIR / "data"
RESULTS_DIR = ROOT / "resultados"
FIGURES_DIR = ROOT / "figuras"

sns.set_theme(style="whitegrid", context="talk")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURES_DIR / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def face_grid(images, titles, h, w, ncols, filename, suptitle=None, centered=False):
    nrows = int(np.ceil(len(images) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 2.9 * nrows))
    axes = np.atleast_1d(axes).ravel()
    if centered:
        scale = np.percentile(np.abs(np.asarray(images)), 99)
        kwargs = {"cmap": "gray", "vmin": -scale, "vmax": scale}
    else:
        kwargs = {"cmap": "gray"}
    for ax, image, title in zip(axes, images, titles):
        ax.imshow(np.asarray(image).reshape(h, w), **kwargs)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    for ax in axes[len(images) :]:
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=16, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, filename)


def main() -> None:
    start = time.perf_counter()
    lfw = fetch_lfw_people(min_faces_per_person=25, data_home=str(DATA_HOME))
    X = np.asarray(lfw.data, dtype=np.float32)
    y = np.asarray(lfw.target)
    names = np.asarray(lfw.target_names)
    n, p = X.shape
    h, w = lfw.images.shape[1:]

    counts = pd.Series(names[y]).value_counts().rename_axis("persona").reset_index(name="imagenes")
    counts.to_csv(RESULTS_DIR / "distribucion_personas.csv", index=False)

    fig, ax = plt.subplots(figsize=(13, 8))
    shown = counts.sort_values("imagenes", ascending=True)
    ax.barh(shown["persona"], shown["imagenes"], color="#26547C")
    ax.set(title="LFW está fuertemente desbalanceada", xlabel="Número de imágenes", ylabel="")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    save_figure(fig, "01_distribucion_personas.png")

    serena_idx = np.flatnonzero(names[y] == "Serena Williams")[:10]
    face_grid(
        X[serena_idx],
        [f"Serena {i + 1}" for i in range(len(serena_idx))],
        h,
        w,
        5,
        "02_serena_variabilidad.png",
        "La misma persona cambia con pose, gesto e iluminación",
    )

    # Partición única antes de aprender cualquier transformación.
    indices = np.arange(n)
    train_idx, test_idx = train_test_split(
        indices, test_size=0.20, random_state=SEED, stratify=y
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    mu_train = X_train.mean(axis=0)
    Xc_train = X_train - mu_train
    Xc_test = X_test - mu_train

    face_grid(
        [mu_train, *Xc_train[:5]],
        ["Promedio train", *[f"Centrada {i + 1}" for i in range(5)]],
        h,
        w,
        3,
        "03_promedio_y_centradas.png",
        "Centrar separa lo común de lo que distingue cada rostro",
        centered=True,
    )

    # SVD truncada hasta el mayor K de interés. El denominador de varianza usa
    # la energía total de Xc_train, no solo la energía de los componentes retenidos.
    U, singular_values, Vt = randomized_svd(
        Xc_train,
        n_components=max(K_GRID),
        n_iter=7,
        random_state=SEED,
    )
    total_energy = float(np.square(Xc_train, dtype=np.float64).sum())
    explained_ratio = np.square(singular_values, dtype=np.float64) / total_energy
    cumulative_ratio = np.cumsum(explained_ratio)

    thresholds = {}
    for threshold in (0.80, 0.90, 0.95):
        reached = np.flatnonzero(cumulative_ratio >= threshold)
        thresholds[str(threshold)] = int(reached[0] + 1) if len(reached) else None

    pd.DataFrame(
        {
            "componente": np.arange(1, len(singular_values) + 1),
            "valor_singular": singular_values,
            "varianza_explicada": explained_ratio,
            "varianza_acumulada": cumulative_ratio,
        }
    ).to_csv(RESULTS_DIR / "varianza_svd_train.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].plot(np.arange(1, 201), singular_values[:200], color="#7B2CBF", lw=2)
    axes[0].set(title="Los valores singulares caen rápidamente", xlabel="Componente", ylabel="Valor singular")
    axes[1].plot(np.arange(1, len(cumulative_ratio) + 1), 100 * cumulative_ratio, color="#0077B6", lw=2)
    for threshold in (80, 90, 95):
        axes[1].axhline(threshold, color="gray", ls="--", lw=1)
    axes[1].set(title="Varianza acumulada en entrenamiento", xlabel="K", ylabel="Varianza capturada (%)", ylim=(0, 100))
    fig.tight_layout()
    save_figure(fig, "04_svd_varianza.png")

    vmax = np.percentile(np.abs(Vt[:12]), 99)
    face_grid(
        Vt[:12],
        [f"Eigenface {i}" for i in range(1, 13)],
        h,
        w,
        4,
        "05_primeras_eigenfaces.png",
        "Eigenfaces: patrones compartidos de variación",
        centered=True,
    )

    # Reconstrucciones y error global por K.
    Z_train_all = Xc_train @ Vt.T
    compression_rows = []
    for k in K_GRID:
        Xhat_train = Z_train_all[:, :k] @ Vt[:k] + mu_train
        rmse = float(np.sqrt(np.mean(np.square(Xhat_train - X_train))))
        compression_rows.append(
            {
                "K": k,
                "varianza_capturada": float(cumulative_ratio[k - 1]),
                "porcentaje_numeros_originales": 100.0 * k / p,
                "reduccion_porcentual": 100.0 * (1.0 - k / p),
                "rmse_reconstruccion_train": rmse,
            }
        )
    compression = pd.DataFrame(compression_rows)
    compression.to_csv(RESULTS_DIR / "compresion_por_K.csv", index=False)

    serena_train_candidates = train_idx[np.isin(train_idx, np.flatnonzero(names[y] == "Serena Williams"))]
    chosen_global = int(serena_train_candidates[0])
    chosen_local = int(np.flatnonzero(train_idx == chosen_global)[0])
    recon_images = [X_train[chosen_local]]
    recon_titles = ["Original\n2.914 valores"]
    for k in K_GRID:
        recon_images.append(mu_train + Z_train_all[chosen_local, :k] @ Vt[:k])
        recon_titles.append(f"K={k}\n{k / p:.1%} del original")
    face_grid(
        recon_images,
        recon_titles,
        h,
        w,
        4,
        "06_reconstrucciones_serena.png",
        "Más K conserva detalle; la clasificación decide si ese detalle es útil",
    )

    # Verificación binaria por usuario y K con el clasificador exigido.
    metric_rows = []
    convergence_notes = []
    for user in PILOT_USERS:
        target_id = int(np.flatnonzero(names == user)[0])
        ytr = (y_train == target_id).astype(int)
        yte = (y_test == target_id).astype(int)
        naive_accuracy = float(1.0 - yte.mean())
        for k in K_GRID:
            model = LogisticRegression(
                solver="sag", random_state=SEED, max_iter=1000
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(Z_train_all[:, :k], ytr)
            convergence_notes.append(
                {
                    "usuario": user,
                    "K": k,
                    "iteraciones": int(model.n_iter_[0]),
                    "convergio": not any(issubclass(w.category, ConvergenceWarning) for w in caught),
                }
            )
            Z_test_k = Xc_test @ Vt[:k].T
            pred = model.predict(Z_test_k)
            tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0, 1]).ravel()
            metric_rows.append(
                {
                    "usuario": user,
                    "K": k,
                    "positivos_train": int(ytr.sum()),
                    "positivos_test": int(yte.sum()),
                    "accuracy_ingenuo": naive_accuracy,
                    "accuracy": accuracy_score(yte, pred),
                    "precision": precision_score(yte, pred, zero_division=0),
                    "recall": recall_score(yte, pred, zero_division=0),
                    "f1": f1_score(yte, pred, zero_division=0),
                    "TN": int(tn),
                    "FP": int(fp),
                    "FN": int(fn),
                    "TP": int(tp),
                }
            )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(RESULTS_DIR / "metricas_clasificacion_por_K.csv", index=False)
    pd.DataFrame(convergence_notes).to_csv(RESULTS_DIR / "convergencia_logistica.csv", index=False)
    ranking = metrics.merge(compression, on="K", how="left")
    ranking.to_csv(RESULTS_DIR / "ranking_configuraciones.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), sharex=True, sharey=True)
    for user, group in metrics.groupby("usuario"):
        short = user.replace("George W ", "G. W. ").replace("Serena ", "S. ")
        axes[0].plot(group["K"], group["precision"], marker="o", label=short)
        axes[1].plot(group["K"], group["recall"], marker="o", label=short)
        axes[2].plot(group["K"], group["f1"], marker="o", label=short)
    for ax, title in zip(axes, ["Precisión", "Recall", "F1"]):
        ax.set(title=title, xlabel="K", ylim=(-0.03, 1.03), xscale="log")
        ax.set_xticks(K_GRID, labels=[str(k) for k in K_GRID])
    axes[0].set_ylabel("Métrica en test")
    axes[2].legend(loc="lower right")
    fig.suptitle("El desempeño depende de K y del usuario piloto", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "07_metricas_por_K.png")

    summary = {
        "n_imagenes": int(n),
        "n_personas": int(len(names)),
        "alto": int(h),
        "ancho": int(w),
        "pixeles_por_rostro": int(p),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "persona_mas_fotografiada": str(counts.iloc[0]["persona"]),
        "max_imagenes_persona": int(counts.iloc[0]["imagenes"]),
        "porcentaje_base_persona_mas_fotografiada": float(100 * counts.iloc[0]["imagenes"] / n),
        "K_para_varianza": thresholds,
        "metodologia": {
            "estandarizacion": "No: todos los píxeles comparten escala de intensidad; escalar por píxel amplificaría regiones de baja varianza.",
            "anti_leakage": "Media, SVD y regresión logística aprendidas solo en train.",
            "svd": "randomized_svd con 500 componentes, n_iter=7 y random_state=10101.",
        },
        "tiempo_segundos": float(time.perf_counter() - start),
    }
    (RESULTS_DIR / "resumen.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nMétricas:\n", metrics.to_string(index=False))
    print(f"\nResultados guardados en: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
