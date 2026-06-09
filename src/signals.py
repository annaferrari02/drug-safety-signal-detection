#import function prr, ror, mgps, bcpnn from notebooks 
"""
src/signals.py

Algoritmi di disproportionality analysis per la farmacovigilanza.
Tutti e quattro gli algoritmi (PRR, ROR, BCPNN, MGPS) partono dalla stessa
contingency table 2x2 prodotta da build_contingency_table() in contingency_table.py.

Schema input comune (pd.DataFrame con colonne):
    drug, pt, a, b, c, d, n

Schema output comune (pd.DataFrame con colonne):
    ae_name, product_name, events, <metriche algoritmo>, signal_positive

La funzione contingency_to_vigipy() converte il formato interno al formato
atteso da vigipy ed è usata internamente da tutti e quattro gli algoritmi.
"""

import numpy as np
import pandas as pd
from vigipy import GPS, bcpnn
from vigipy.PRR import prr
from vigipy.ROR import ror
from vigipy.utils import Container


# PRIOR EMPIRICI DI DUMOUCHEL (1999)
# Stimati sull'intero database FAERS nel paper originale che ha introdotto il GPS.
# Usati come fallback quando la matrice di contingenza ha troppo poche righe
# per stimare i prior in modo affidabile (es. analisi su singolo drug, strati piccoli).
DUMOUCHEL_PRIORS = [0.2041, 0.05816, 1.415, 1.838, 0.0969]


def contingency_to_vigipy(ct: pd.DataFrame) -> tuple[Container, int]:
    """
    Converte la contingency table 2x2 nel formato Container atteso da vigipy.

    Mapping colonne:
        drug  → product_name
        pt    → ae_name
        a     → events          (co-occorrenze drug × PT)
        a+b   → product_aes     (tutti i report con questo drug)
        a+c   → count_across_brands  (tutti i report con questo PT)

    La matrice pivot (drug × PT) viene costruita e allegata al container:
        vigipy la usa internamente per stimare i prior bayesiani (MGPS, BCPNN).

    Parameters
    ----------
    ct : pd.DataFrame
        Output di build_contingency_table() con colonne [drug, pt, a, b, c, d, n].

    Returns
    -------
    container : Container  — oggetto vigipy pronto per gli algoritmi
    N         : int        — totale report nel background (da ct["n"].iloc[0])
    """
    df = ct.copy()
    df = df.rename(columns={"drug": "product_name", "pt": "ae_name", "a": "events"})
    df["product_aes"]         = df["events"] + df["b"]  # a+b: report con questo drug
    df["count_across_brands"] = df["events"] + df["c"]  # a+c: report con questo PT

    N = int(df["n"].iloc[0])  # n è costante per tutte le righe della stessa CT

    container = Container(params=False)
    container.data = df[["product_name", "ae_name", "events",
                          "product_aes", "count_across_brands"]]
    container.N = N

    # Matrice pivot drug × PT: serve a vigipy per stimare i prior bayesiani
    container.contingency = df.pivot_table(
        index="product_name",
        columns="ae_name",
        values="events",
        fill_value=0,
    )

    return container, N


def compute_prr(
        ct:            pd.DataFrame,
        min_events:    int   = 3,
        decision_thres: float = 0.05,
) -> pd.DataFrame:
    """
    Proportional Reporting Ratio (PRR).

    Criterio di positività del segnale: FDR < decision_thres (default 0.05).
    Metrica di ranking: p_value.

    Parameters
    ----------
    ct             : contingency table da build_contingency_table()
    min_events     : soglia minima sulla cella a, default 3
    decision_thres : soglia FDR per dichiarare un segnale positivo, default 0.05

    Returns
    -------
    pd.DataFrame con colonne:
        ae_name, product_name, events, PRR, PRR_lower_bound, PRR_upper_bound,
        p_value, fdr, signal_positive
    """
    container, _ = contingency_to_vigipy(ct)

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
        "Product":       "product_name",
        "Adverse Event": "ae_name",
        "Count":         "events",
    })


def compute_ror(
        ct:            pd.DataFrame,
        min_events:    int   = 3,
        decision_thres: float = 0.05,
) -> pd.DataFrame:
    """
    Reporting Odds Ratio (ROR).

    Criterio di positività del segnale: FDR < decision_thres (default 0.05).
    Metrica di ranking: p_value.

    Parameters
    ----------
    ct             : contingency table da build_contingency_table()
    min_events     : soglia minima sulla cella a, default 3
    decision_thres : soglia FDR per dichiarare un segnale positivo, default 0.05

    Returns
    -------
    pd.DataFrame con colonne:
        ae_name, product_name, events, ROR, ROR_lower_bound, ROR_upper_bound,
        p_value, fdr, signal_positive
    """
    container, _ = contingency_to_vigipy(ct)

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
        "Product":       "product_name",
        "Adverse Event": "ae_name",
        "Count":         "events",
    })


def compute_bcpnn(
        ct:         pd.DataFrame,
        min_events: int = 3,
        ic_threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Bayesian Confidence Propagation Neural Network (BCPNN).

    Metrica principale: IC (Information Component) = quantile (lower bound 95% CI).
    Criterio di positività del segnale: IC > ic_threshold (default 0, come da
    Cerbito et al. 2026 e standard WHO-UMC).

    Parameters
    ----------
    ct           : contingency table da build_contingency_table()
    min_events   : soglia minima sulla cella a, default 3
    ic_threshold : soglia IC025 per dichiarare un segnale positivo, default 0.0

    Returns
    -------
    pd.DataFrame con colonne:
        ae_name, product_name, events, IC, IC_lower_bound (quantile),
        count_expected, signal_positive
    """
    container, _ = contingency_to_vigipy(ct)

    results = bcpnn(
        container=container,
        min_events=min_events,
        decision_metric="rank",
        ranking_statistic="quantile",  # quantile = IC025 (lower bound 95% CI)
    )

    df = results.all_signals.copy()
    # IC025 > 0 è il criterio standard: il lower bound del CI è sopra la soglia di indifferenza
    df["signal_positive"] = df["quantile"] > ic_threshold

    return df.rename(columns={
        "Product":        "product_name",
        "Adverse Event":  "ae_name",
        "Count":          "events",
        "Expected Count": "count_expected",
        "quantile":       "IC_lower_bound",
        "count/expected": "IC",
    })


def compute_mgps(
        ct:           pd.DataFrame,
        min_events:   int   = 3,
        eb05_threshold: float = 2.0,
        force_priors: bool  = True,
) -> pd.DataFrame:
    """
    Multi-item Gamma Poisson Shrinker (MGPS / EBGM).

    Metrica principale: EBGM (Empirical Bayes Geometric Mean) = log2.
    Criterio di positività del segnale: EB05 >= eb05_threshold (default 2.0),
    dove EB05 è il lower bound al 90% CI dell'EBGM.

    Parameters
    ----------
    ct             : contingency table da build_contingency_table()
    min_events     : soglia minima sulla cella a, default 3
    eb05_threshold : soglia EB05 per dichiarare un segnale positivo, default 2.0
    force_priors   : se True, usa sempre i prior DuMouchel 1999 invece di stimarli
                     dalla matrice. Raccomandato per analisi su singolo drug o
                     strati piccoli dove la stima dei prior è instabile.

    Returns
    -------
    pd.DataFrame con colonne:
        ae_name, product_name, events, EBGM, EB05, count_expected,
        p_value, signal_positive
    """
    container, _ = contingency_to_vigipy(ct)

    kwargs = dict(
        relative_risk=1,
        min_events=min_events,
        decision_metric="rank",
        ranking_statistic="log2",        # log2 = EBGM
        expected_method="mantel-haentzel",
    )

    if force_priors:
        kwargs["prior_param"] = DUMOUCHEL_PRIORS

    results = GPS.gps(container, **kwargs)

    df = results.all_signals.copy()
    df["signal_positive"] = df["LowerBound"] >= eb05_threshold

    return df.rename(columns={
        "Product":        "product_name",
        "Adverse Event":  "ae_name",
        "Count":          "events",
        "Expected Count": "count_expected",
        "log2":           "EBGM",
        "LowerBound":     "EB05",
    })