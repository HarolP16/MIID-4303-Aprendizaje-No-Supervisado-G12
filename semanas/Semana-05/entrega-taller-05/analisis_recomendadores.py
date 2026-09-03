"""Análisis reproducible del Taller 5: módulo Descubre.

Implementa cinco motores sobre consumo implícito de Last.fm y los compara con
una partición holdout por usuario. Los ceros se interpretan como desconocidos.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

RANDOM_STATE = 123
TOP_N = 10
K_NEIGHBORS = 50
SVD_CANDIDATES = (10, 25, 50, 100)
MIN_SUPPORT_COUNT = 5
MIN_CONFIDENCE = 0.05
MIN_LIFT = 1.20

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "Taller semana5"
RESULTS_DIR = HERE / "resultados"
FIGURES_DIR = HERE / "figuras"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    artists = pd.read_csv(DATA_DIR / "artists.txt", sep="\t")
    interactions = pd.read_csv(DATA_DIR / "user_artists.txt", sep="\t")
    artists = artists.rename(columns={"id": "artistID", "name": "artist_name"})
    merged = interactions.merge(
        artists[["artistID", "artist_name"]], on="artistID", how="left", validate="many_to_one"
    )
    return artists, interactions, merged


def make_maps(artists, interactions):
    user_ids = np.sort(interactions["userID"].unique())
    artist_ids = np.sort(artists["artistID"].unique())
    user_to_idx = {int(v): i for i, v in enumerate(user_ids)}
    artist_to_idx = {int(v): i for i, v in enumerate(artist_ids)}
    idx_to_artist = dict(enumerate(artist_ids.astype(int)))
    names = artists.set_index("artistID")["artist_name"].to_dict()
    return user_ids, artist_ids, user_to_idx, artist_to_idx, idx_to_artist, names


def stratified_holdout(interactions, fraction=0.20):
    rng = np.random.default_rng(RANDOM_STATE)
    train_parts, test_parts = [], []
    for _, group in interactions.groupby("userID", sort=True):
        n_test = max(1, int(round(len(group) * fraction)))
        n_test = min(n_test, len(group) - 1)
        selected = rng.choice(group.index.to_numpy(), size=n_test, replace=False)
        test_parts.append(group.loc[selected])
        train_parts.append(group.drop(selected))
    return pd.concat(train_parts).sort_index(), pd.concat(test_parts).sort_index()


def build_matrix(df, user_to_idx, artist_to_idx, n_users, n_artists, value="log"):
    rows = df["userID"].map(user_to_idx).to_numpy()
    cols = df["artistID"].map(artist_to_idx).to_numpy()
    if value == "binary":
        vals = np.ones(len(df), dtype=np.float32)
    elif value == "log":
        vals = np.log1p(df["weight"].to_numpy()).astype(np.float32)
    else:
        vals = df["weight"].to_numpy(dtype=np.float32)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n_users, n_artists))


def top_indices(scores, seen, n=TOP_N):
    values = np.asarray(scores, dtype=float).ravel().copy()
    values[np.asarray(seen, dtype=int)] = -np.inf
    finite = np.flatnonzero(np.isfinite(values))
    if finite.size <= n:
        order = finite[np.argsort(values[finite])[::-1]]
    else:
        part = np.argpartition(values, -n)[-n:]
        order = part[np.argsort(values[part])[::-1]]
    return order[:n]


def recommend_global(score_vector, seen_lists):
    return [top_indices(score_vector, seen) for seen in seen_lists]


def fit_cosine(matrix, k=K_NEIGHBORS):
    unit = normalize(matrix, norm="l2", axis=1)
    sim = (unit @ unit.T).toarray().astype(np.float32)
    np.fill_diagonal(sim, 0.0)
    if k < sim.shape[1]:
        cut = np.argpartition(sim, -k, axis=1)[:, :-k]
        np.put_along_axis(sim, cut, 0.0, axis=1)
    denom = sim.sum(axis=1, keepdims=True)
    sim = np.divide(sim, denom, out=np.zeros_like(sim), where=denom > 0)
    scores = sparse.csr_matrix(sim) @ unit
    return sim, scores.tocsr()


def recommend_sparse_rows(score_matrix, seen_lists):
    recs = []
    for u in range(score_matrix.shape[0]):
        row = score_matrix.getrow(u).toarray().ravel()
        recs.append(top_indices(row, seen_lists[u]))
    return recs


def recommend_svd(matrix, seen_lists, k, return_model=False):
    model = TruncatedSVD(n_components=k, random_state=RANDOM_STATE, n_iter=7)
    latent = model.fit_transform(matrix)
    recs = []
    for start in range(0, matrix.shape[0], 128):
        end = min(start + 128, matrix.shape[0])
        batch_scores = latent[start:end] @ model.components_
        for local, u in enumerate(range(start, end)):
            recs.append(top_indices(batch_scores[local], seen_lists[u]))
    return (recs, model, latent) if return_model else recs


def build_apriori_rules(binary_matrix):
    supports = np.asarray(binary_matrix.sum(axis=0)).ravel().astype(int)
    frequent = set(np.flatnonzero(supports >= MIN_SUPPORT_COUNT).tolist())
    pair_counts = Counter()
    for u in range(binary_matrix.shape[0]):
        basket = [int(i) for i in binary_matrix.indices[binary_matrix.indptr[u]:binary_matrix.indptr[u + 1]] if i in frequent]
        basket.sort()
        for i, left in enumerate(basket):
            for right in basket[i + 1:]:
                pair_counts[(left, right)] += 1

    rules = defaultdict(list)
    n_users = binary_matrix.shape[0]
    rows = []
    for (left, right), count in pair_counts.items():
        if count < MIN_SUPPORT_COUNT:
            continue
        pair_support = count / n_users
        for antecedent, consequent in ((left, right), (right, left)):
            confidence = count / supports[antecedent]
            lift = confidence / (supports[consequent] / n_users)
            if confidence >= MIN_CONFIDENCE and lift >= MIN_LIFT:
                score = confidence * math.log2(lift)
                rules[antecedent].append((consequent, score, pair_support, confidence, lift))
                rows.append((antecedent, consequent, count, pair_support, confidence, lift, score))
    for antecedent in rules:
        rules[antecedent].sort(key=lambda x: (x[1], x[3], x[4]), reverse=True)
    columns = ["antecedent_idx", "consequent_idx", "pair_count", "support", "confidence", "lift", "rule_score"]
    return rules, pd.DataFrame(rows, columns=columns)


def recommend_apriori(rules, seen_lists, popularity_fallback=None):
    recs = []
    fallback_count = 0
    for seen in seen_lists:
        seen_set = set(map(int, seen))
        candidates = defaultdict(float)
        for antecedent in seen_set:
            for consequent, score, *_ in rules.get(antecedent, []):
                if consequent not in seen_set:
                    candidates[consequent] += score
        ranked = [i for i, _ in sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:TOP_N]]
        if len(ranked) < TOP_N and popularity_fallback is not None:
            fallback_count += 1
            forbidden = seen_set | set(ranked)
            for idx in np.argsort(popularity_fallback)[::-1]:
                if idx not in forbidden:
                    ranked.append(int(idx))
                    if len(ranked) == TOP_N:
                        break
        recs.append(np.asarray(ranked[:TOP_N], dtype=int))
    return recs, fallback_count


def evaluate(recs, test_lists, train_support, catalog_size, rng):
    recalls = []
    reciprocal_ranks = []
    all_recs = []
    novelties = []
    long_tail_cut = np.quantile(train_support[train_support > 0], 0.80)
    long_tail = []
    n_users = len(test_lists)
    for rec, truth in zip(recs, test_lists):
        rec = np.asarray(rec, dtype=int)
        truth_set = set(map(int, truth))
        hits = [rank for rank, item in enumerate(rec, start=1) if int(item) in truth_set]
        recalls.append(len(hits) / max(len(truth_set), 1))
        reciprocal_ranks.append(0 if not hits else 1 / min(hits))
        all_recs.extend(rec.tolist())
        novelties.extend((-np.log2((train_support[rec] + 1) / (n_users + 1))).tolist())
        long_tail.extend((train_support[rec] <= long_tail_cut).tolist())

    pair_scores = []
    for _ in range(5000):
        a, b = rng.integers(0, len(recs), size=2)
        if a == b:
            continue
        sa, sb = set(map(int, recs[a])), set(map(int, recs[b]))
        pair_scores.append(len(sa & sb) / max(len(sa | sb), 1))
    return {
        "recall_at_10": float(np.mean(recalls)),
        "mrr_at_10": float(np.mean(reciprocal_ranks)),
        "catalog_coverage": len(set(all_recs)) / catalog_size,
        "novelty_bits": float(np.mean(novelties)),
        "long_tail_share": float(np.mean(long_tail)),
        "personalization": 1 - float(np.mean(pair_scores)),
        "unique_artists_recommended": len(set(all_recs)),
    }


def save_recommendations(recs_by_model, user_ids, idx_to_artist, names, target_users):
    rows = []
    for model_name, recs in recs_by_model.items():
        for user_id in target_users:
            uidx = int(np.flatnonzero(user_ids == user_id)[0])
            for rank, item_idx in enumerate(recs[uidx], start=1):
                artist_id = idx_to_artist[int(item_idx)]
                rows.append((user_id, model_name, rank, artist_id, names.get(artist_id, "(sin nombre)")))
    out = pd.DataFrame(rows, columns=["userID", "motor", "rank", "artistID", "artist_name"])
    out.to_csv(RESULTS_DIR / "recomendaciones_usuarios.csv", index=False, encoding="utf-8-sig")
    return out


def minmax(series):
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(np.ones(len(series)), index=series.index)
    return (series - lo) / (hi - lo)


def main():
    sns.set_theme(style="whitegrid", context="talk")
    artists, interactions, merged = load_data()
    user_ids, artist_ids, user_to_idx, artist_to_idx, idx_to_artist, names = make_maps(artists, interactions)
    n_users, n_artists = len(user_ids), len(artist_ids)

    artist_stats = merged.groupby(["artistID", "artist_name"], as_index=False).agg(
        unique_listeners=("userID", "nunique"), total_plays=("weight", "sum"), median_plays=("weight", "median")
    )
    artist_stats["log_weighted_popularity"] = merged.assign(log_weight=np.log1p(merged["weight"])).groupby("artistID")["log_weight"].sum().reindex(artist_stats["artistID"]).to_numpy()
    artist_stats["rank_listeners"] = artist_stats["unique_listeners"].rank(method="min", ascending=False).astype(int)
    artist_stats["rank_plays"] = artist_stats["total_plays"].rank(method="min", ascending=False).astype(int)
    artist_stats["rank_change"] = artist_stats["rank_listeners"] - artist_stats["rank_plays"]
    artist_stats.sort_values("unique_listeners", ascending=False).to_csv(RESULTS_DIR / "popularidad_artistas.csv", index=False, encoding="utf-8-sig")

    user8 = merged.query("userID == 8").copy()
    user8["relative_plays"] = user8["weight"] / user8["weight"].sum()
    user8["cumulative_share"] = user8.sort_values("relative_plays", ascending=False)["relative_plays"].cumsum()
    user8.sort_values("relative_plays", ascending=False).to_csv(RESULTS_DIR / "perfil_usuario_8.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    stats_sorted = artist_stats.sort_values("unique_listeners", ascending=False).reset_index(drop=True)
    axes[0].loglog(np.arange(1, len(stats_sorted) + 1), stats_sorted["unique_listeners"])
    axes[0].set(title="Cola larga del catálogo", xlabel="Rango del artista", ylabel="Usuarios únicos")
    u8_sorted = user8.sort_values("relative_plays", ascending=False).head(15)
    sns.barplot(data=u8_sorted, y="artist_name", x="relative_plays", ax=axes[1], color="#04A6C2")
    axes[1].set(title="Usuario 8: concentración de reproducciones", xlabel="Proporción de sus reproducciones", ylabel="")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "01_estructura_consumo.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    train, test = stratified_holdout(interactions)
    train_binary = build_matrix(train, user_to_idx, artist_to_idx, n_users, n_artists, "binary")
    train_log = build_matrix(train, user_to_idx, artist_to_idx, n_users, n_artists, "log")
    train_model = normalize(train_log, norm="l2", axis=1)
    seen_lists = [train_binary.indices[train_binary.indptr[u]:train_binary.indptr[u + 1]] for u in range(n_users)]
    full_binary = build_matrix(interactions, user_to_idx, artist_to_idx, n_users, n_artists, "binary")
    full_seen_lists = [full_binary.indices[full_binary.indptr[u]:full_binary.indptr[u + 1]] for u in range(n_users)]
    test_lists = [test.loc[test["userID"] == uid, "artistID"].map(artist_to_idx).to_numpy() for uid in user_ids]
    support = np.asarray(train_binary.sum(axis=0)).ravel()
    weighted = np.asarray(train_log.sum(axis=0)).ravel()

    recs = {}
    recs["Popularidad simple"] = recommend_global(support, seen_lists)
    recs["Popularidad ponderada"] = recommend_global(weighted, seen_lists)
    _, cosine_scores = fit_cosine(train_model)
    recs["Coseno usuario-usuario"] = recommend_sparse_rows(cosine_scores, seen_lists)

    svd_metrics = []
    svd_recs = {}
    for k in SVD_CANDIDATES:
        candidate_recs = recommend_svd(train_model, seen_lists, k)
        score = evaluate(candidate_recs, test_lists, support, n_artists, np.random.default_rng(RANDOM_STATE + k))
        score["k"] = k
        svd_metrics.append(score)
        svd_recs[k] = candidate_recs
    svd_selection = pd.DataFrame(svd_metrics).sort_values(["recall_at_10", "catalog_coverage"], ascending=False)
    selected_k = int(svd_selection.iloc[0]["k"])
    recs["SVD"] = svd_recs[selected_k]
    svd_selection.to_csv(RESULTS_DIR / "seleccion_k_svd.csv", index=False)

    apriori_rules, rules_df = build_apriori_rules(train_binary)
    recs["Apriori"] , apriori_fallbacks = recommend_apriori(apriori_rules, seen_lists, weighted)

    explainability = {
        "Popularidad simple": 1.00,
        "Popularidad ponderada": 1.00,
        "Coseno usuario-usuario": 0.85,
        "SVD": 0.45,
        "Apriori": 1.00,
    }
    metric_rows = []
    for model_name, model_recs in recs.items():
        row = evaluate(model_recs, test_lists, support, n_artists, np.random.default_rng(RANDOM_STATE))
        row["motor"] = model_name
        row["explainability"] = explainability[model_name]
        metric_rows.append(row)
    metrics = pd.DataFrame(metric_rows).set_index("motor")
    weights = {
        "recall_at_10": 0.35,
        "catalog_coverage": 0.20,
        "novelty_bits": 0.15,
        "personalization": 0.15,
        "explainability": 0.15,
    }
    metrics["business_score"] = sum(weights[col] * minmax(metrics[col]) for col in weights)
    metrics["rank"] = metrics["business_score"].rank(method="min", ascending=False).astype(int)
    metrics = metrics.sort_values("rank")
    metrics.to_csv(RESULTS_DIR / "ranking_motores.csv", encoding="utf-8-sig")

    # Recomendaciones ilustrativas excluyen todo lo ya escuchado, incluso el holdout.
    display_recs = {
        "Popularidad simple": recommend_global(support, full_seen_lists),
        "Popularidad ponderada": recommend_global(weighted, full_seen_lists),
    }
    display_recs["Coseno usuario-usuario"] = [top_indices(cosine_scores.getrow(u).toarray().ravel(), full_seen_lists[u]) for u in range(n_users)]
    selected_model = TruncatedSVD(n_components=selected_k, random_state=RANDOM_STATE, n_iter=7)
    latent = selected_model.fit_transform(train_model)
    display_svd = []
    for start in range(0, n_users, 128):
        scores = latent[start:start + 128] @ selected_model.components_
        for local, u in enumerate(range(start, min(start + 128, n_users))):
            display_svd.append(top_indices(scores[local], full_seen_lists[u]))
    display_recs["SVD"] = display_svd
    display_recs["Apriori"], _ = recommend_apriori(apriori_rules, full_seen_lists, weighted)
    target_users = [8, int(user_ids[0]), int(user_ids[len(user_ids) // 3]), int(user_ids[2 * len(user_ids) // 3])]
    target_users = list(dict.fromkeys(target_users))
    recommendations = save_recommendations(display_recs, user_ids, idx_to_artist, names, target_users)

    rules_df["antecedent_artistID"] = rules_df["antecedent_idx"].map(idx_to_artist)
    rules_df["consequent_artistID"] = rules_df["consequent_idx"].map(idx_to_artist)
    rules_df["antecedent"] = rules_df["antecedent_artistID"].map(names)
    rules_df["consequent"] = rules_df["consequent_artistID"].map(names)
    rules_df.sort_values("rule_score", ascending=False).head(1000).to_csv(RESULTS_DIR / "reglas_apriori_top.csv", index=False, encoding="utf-8-sig")

    factor_rows = []
    for factor, component in enumerate(selected_model.components_[: min(selected_k, 5)], start=1):
        for side, indices in (("positivo", np.argsort(component)[-10:][::-1]), ("negativo", np.argsort(component)[:10])):
            for rank, idx in enumerate(indices, start=1):
                artist_id = idx_to_artist[int(idx)]
                factor_rows.append((factor, side, rank, artist_id, names.get(artist_id), float(component[idx])))
    pd.DataFrame(factor_rows, columns=["factor", "lado", "rank", "artistID", "artist_name", "loading"]).to_csv(RESULTS_DIR / "cargas_factores_svd.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_metrics = metrics.sort_values("business_score")
    axes[0].barh(plot_metrics.index, plot_metrics["business_score"], color="#04A6C2")
    axes[0].set(xlabel="Puntaje ponderado [0,1]", title="Ranking para el objetivo Descubre", xlim=(0, 1))
    scatter = axes[1].scatter(metrics["recall_at_10"], metrics["catalog_coverage"], s=metrics["novelty_bits"] ** 2 * 6, c=metrics["business_score"], cmap="viridis")
    for name, row in metrics.iterrows():
        axes[1].annotate(name.replace(" usuario-usuario", ""), (row["recall_at_10"], row["catalog_coverage"]), xytext=(5, 5), textcoords="offset points", fontsize=9)
    axes[1].set(xlabel="Recall@10", ylabel="Cobertura del catálogo", title="Precisión y descubrimiento entran en tensión")
    fig.colorbar(scatter, ax=axes[1], label="Puntaje de negocio")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "02_ranking_motores.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    u8_recs = recommendations.query("userID == 8")
    pivot = u8_recs.pivot(index="rank", columns="motor", values="artist_name")
    pivot.to_csv(RESULTS_DIR / "recomendaciones_usuario_8_comparadas.csv", encoding="utf-8-sig")

    observed_pairs = len(interactions)
    density = observed_pairs / (n_users * n_artists)
    singletons = int((artist_stats["unique_listeners"] == 1).sum())
    summary = {
        "random_state": RANDOM_STATE,
        "top_n": TOP_N,
        "n_users": n_users,
        "n_artists_catalog": n_artists,
        "n_artists_observed": int(interactions["artistID"].nunique()),
        "n_pairs": observed_pairs,
        "density": density,
        "median_weight": float(interactions["weight"].median()),
        "max_weight": int(interactions["weight"].max()),
        "single_listener_artists": singletons,
        "single_listener_share_catalog": singletons / n_artists,
        "selected_svd_k": selected_k,
        "apriori_rules": int(len(rules_df)),
        "apriori_users_with_popularity_fallback": int(apriori_fallbacks),
        "recommended_engine": metrics.index[0],
        "ranking_weights": weights,
        "limitations": [
            "No hay marcas de tiempo: el holdout aleatorio no simula una evaluación futura.",
            "Ausencia significa desconocido porque cada perfil está truncado a sus 50 artistas principales.",
            "Los datos de 2011 sirven para comparar motores, no para fijar artistas actuales.",
        ],
    }
    (RESULTS_DIR / "resumen.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nRanking:\n", metrics.round(4).to_string())
    print("\nUsuario 8:\n", pivot.to_string())


if __name__ == "__main__":
    main()
