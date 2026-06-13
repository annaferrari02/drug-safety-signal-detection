import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
from src.text_to_sql import parse_nl_to_params, nl_to_contingency_table

st.title("Drug Safety Signal Detection")
st.markdown("Fai una domanda in linguaggio naturale sui dati FAERS.")

user_input = st.text_input(
    "La tua domanda:",
    placeholder="es: analizza gli effetti avversi di LAPATINIB nelle donne anziane"
)

if st.button("Esegui"):
    if user_input:

        # 1. Mostra i parametri estratti da Mistral
        with st.spinner("Estraggo i parametri..."):
            params = parse_nl_to_params(user_input)

        st.subheader("Parametri estratti")
        col1, col2, col3 = st.columns(3)
        col1.metric("Drug", params["target_drug"])
        col2.metric("Min occorrenze (a)", params.get("min_a", 3))
        col3.metric("Filtro", params.get("where_extra") or "Nessuno")

        # 2. Esegui la contingency table
        with st.spinner("Costruisco la contingency table..."):
            try:
                df = nl_to_contingency_table(user_input)

                st.subheader(f"Risultati — {len(df)} coppie (drug, PT)")
                st.dataframe(df, use_container_width=True)

                # 3. Download CSV
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Scarica CSV",
                    data=csv,
                    file_name=f"{params['target_drug']}_contingency.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"Errore durante l'esecuzione: {e}")
    else:
        st.warning("Inserisci una domanda prima di eseguire.")
