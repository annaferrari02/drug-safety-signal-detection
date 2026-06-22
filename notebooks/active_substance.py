import json
import requests


def estrai_principi_attivi_con_key(api_key, limite=1000):
    # Endpoint openFDA con il conteggio dei principi attivi univoci
    url = f"https://api.fda.gov/drug/event.json?api_key={api_key}&count=patient.drug.openfda.substance_name.exact&limit={limite}"

    try:
        print("Interrogazione di OpenFDA con API Key in corso...")
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        # Usiamo un set per garantire l'assoluta univocità (case-insensitive se necessario)
        principi_unici = set()

        if "results" in data:
            for result in data["results"]:
                # openFDA restituisce i nomi in MAIUSCOLO, li teniamo puliti
                nome_sostanza = result["term"].strip()
                if nome_sostanza:
                    principi_unici.add(nome_sostanza)

        # Trasformiamo in una lista ordinata alfabeticamente
        return sorted(list(principi_unici))

    except requests.exceptions.RequestException as e:
        print(f"Errore durante la richiesta: {e}")
        return []


def salva_in_json(lista_sostanze, nome_file="principi_attivi.json"):
    try:
        with open(nome_file, "w", encoding="utf-8") as f:
            # Salva la lista in formato JSON leggibile (indentato)
            json.dump(lista_sostanze, f, ensure_ascii=False, indent=4)
        print(f"\n[OK] File '{nome_file}' generato con successo!")
    except IOError as e:
        print(f"Errore durante il salvataggio del file: {e}")


if __name__ == "__main__":
    MIA_KEY = "PXWQGsdec5W8FbwJJjyGqftn6y1yaJPPfX6KbNqP"

    # Estraiamo i principi attivi (fino a 1000)
    sostanze = estrai_principi_attivi_con_key(api_key=MIA_KEY, limite=1000)

    if sostanze:
        print(f"Estratti {len(sostanze)} principi attivi univoci.")

        # Genera il file JSON
        salva_in_json(sostanze)
    else:
        print("Nessun dato estratto. Controlla l'API Key o la connessione.")