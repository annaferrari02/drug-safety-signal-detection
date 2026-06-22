#import function prr, ror, mgps, bcpnn from notebooks 
"""
src/signals.py — versione con fix PRR/ROR

Fix applicato: in contingency_to_vigipy(), la matrice pivot viene
forzata a dtype float64 con .fillna(0).astype(float).
Senza questo, vigipy riceve una matrice con dtype object quando
ci sono NaN, causando "setting an array element with a sequence"
in PRR e ROR (ma non in BCPNN/MGPS che hanno una gestione diversa).
"""

import numpy as np
import pandas as pd
from vigipy import GPS, bcpnn
from vigipy.PRR import prr
from vigipy.ROR import ror
from vigipy.utils import Container

DUMOUCHEL_PRIORS = [0.2041, 0.05816, 1.415, 1.838, 0.0969]


def contingency_to_vigipy(ct: pd.DataFrame) -> tuple[Container, int]:
    df = ct.copy()
    df = df.rename(columns={"drug": "product_name", "pt": "ae_name", "a": "events"})
    df["product_aes"]         = df["events"] + df["b"]
    df["count_across_brands"] = df["events"] + df["c"]

    # Forza dtype numerico su tutte le colonne numeriche — evita che
    # colonne con dtype object (dalla route cubo OLAP) causino
    # "setting an array element with a sequence" in np.asarray() dentro PRR/ROR
    for col in ["events", "product_aes", "count_across_brands", "b", "c", "d", "n"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(float)

    N = int(df["n"].iloc[0])

    container = Container(params=False)
    container.data = df[["product_name", "ae_name", "events",
                          "product_aes", "count_across_brands"]]
    container.N = N
    container.contingency = df.pivot_table(
        index="product_name",
        columns="ae_name",
        values="events",
        fill_value=0,
    ).fillna(0).astype(float)

    return container, N


def compute_prr(ct, min_events=3, decision_thres=0.05):
    container, _ = contingency_to_vigipy(ct)
    if container is None:
        return pd.DataFrame()
    results = prr(
        container,
        relative_risk=1,
        min_events=min_events,
        decision_metric="fdr",
        decision_thres=decision_thres,
        ranking_statistic="p_value",
        expected_method="mantel-haentzel",
    )
    df = results.all_signals.copy()
    df["signal_positive"] = df["fdr"] < decision_thres
    return df.rename(columns={
        "Product": "product_name", "Adverse Event": "ae_name", "Count": "events",
    })


def compute_ror(ct, min_events=3, decision_thres=0.05):
    container, _ = contingency_to_vigipy(ct)
    if container is None:
        return pd.DataFrame()
    results = ror(
        container,
        relative_risk=1,
        min_events=min_events,
        decision_metric="fdr",
        decision_thres=decision_thres,
        ranking_statistic="p_value",
        expected_method="mantel-haentzel",
    )
    df = results.all_signals.copy()
    df["signal_positive"] = df["fdr"] < decision_thres
    return df.rename(columns={
        "Product": "product_name", "Adverse Event": "ae_name", "Count": "events",
    })


def compute_bcpnn(ct, min_events=3, ic_threshold=0.0):
    container, _ = contingency_to_vigipy(ct)
    results = bcpnn(
        container=container,
        min_events=min_events,
        decision_metric="rank",
        ranking_statistic="quantile",
    )
    df = results.all_signals.copy()
    df["signal_positive"] = df["quantile"] > ic_threshold
    return df.rename(columns={
        "Product": "product_name", "Adverse Event": "ae_name", "Count": "events",
        "Expected Count": "count_expected", "quantile": "IC_lower_bound",
        "count/expected": "IC",
    })


def compute_mgps(ct, min_events=3, eb05_threshold=2.0, force_priors=True):
    container, _ = contingency_to_vigipy(ct)
    kwargs = dict(
        relative_risk=1,
        min_events=min_events,
        decision_metric="rank",
        ranking_statistic="log2",
        expected_method="mantel-haentzel",
    )
    if force_priors:
        kwargs["prior_param"] = DUMOUCHEL_PRIORS
    results = GPS.gps(container, **kwargs)
    df = results.all_signals.copy()
    df["signal_positive"] = df["LowerBound"] >= eb05_threshold
    return df.rename(columns={
        "Product": "product_name", "Adverse Event": "ae_name", "Count": "events",
        "Expected Count": "count_expected", "log2": "EBGM", "LowerBound": "EB05",
    })