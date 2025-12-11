# Previsione delle vendite settimanali dei negozi Walmart

> Un progetto di Machine Learning per prevedere le vendite settimanali dei negozi Walmart. Confronta l'efficacia di un modello di **Regressione Lineare** e di una **Rete Neurale (MLP)**, con un'analisi dettagliata sull'impatto delle festività e della stagionalità sui ricavi.

## Obiettivo del Progetto

L'obiettivo principale di questo progetto è sviluppare un modello robusto per stimare le **Vendite Settimanali (Weekly\_Sales)** di 45 diversi negozi Walmart.
L'analisi mira a identificare i fattori chiave che influenzano le vendite, dando particolare risalto alla **stagionalità** e all'impatto delle **variabili macroeconomiche** come l'inflazione (CPI), il prezzo del carburante e il tasso di disoccupazione.

## Caratteristiche Principali

Il file `Prediction_model.py` implementa il seguente flusso di lavoro di Machine Learning:

* **Data Engineering:** Conversione della `Date` e creazione di feature temporali (`WeekOfYear`, `Month`).
* **Feature Encoding:** Codifica One-Hot per la variabile categoriale `Store`.
* **Standardizzazione:** Utilizzo di `StandardScaler` sulle feature continue per ottimizzare l'addestramento dei modelli, in particolare l'MLP.
* **Modelli Confrontati:**
    * **Regressione Lineare:** Utilizzato come benchmark.
    * **MLP Regressor (Rete Neurale):** Per catturare relazioni più complesse e non lineari.
* **Analisi Esplorativa (EDA):** Visualizzazione dell'andamento stagionale delle vendite.

## Dati Utilizzati

Il progetto utilizza il dataset `Walmart_Store_sales.csv`, contenente vendite settimanali e dati economici rilevanti dal 2010 al 2012.

| Colonna | Tipo | Ruolo |
| :--- | :--- | :--- |
| **Weekly\_Sales** | Float | **Target** (Vendite Settimanali). |
| **Store** | Int | Variabile categoriale per l'ID del negozio. |
| **Holiday\_Flag** | Int | 1 se è una settimana di festa, 0 altrimenti. |
| **CPI** | Float | Indice dei Prezzi al Consumo (Inflazione). |
| **Unemployment** | Float | Tasso di disoccupazione. |

## Installazione

Per eseguire lo script sul tuo sistema, è necessario avere installato Python e le librerie elencate.

### Prerequisiti

* Python 3.x
* `pandas`
* `numpy`
* `scikit-learn`
* `matplotlib`

### Istruzioni

1.  Clona il repository GitHub:
    ```bash
    git clone [https://github.com/biondomarisol/Walmart-Sales-Prediction-Project.git](https://github.com/biondomarisol/Walmart-Sales-Prediction-Project.git)
    ```
2.  Vai alla directory del progetto:
    ```bash
    cd Walmart-Sales-Prediction-Project
    ```
3.  Installa le dipendenze richieste:
    ```bash
    pip install pandas numpy scikit-learn matplotlib
    ```

## Utilizzo

Avvia lo script Python per eseguire l'intero processo di analisi, addestramento e valutazione:

```bash
python Prediction_model.py
