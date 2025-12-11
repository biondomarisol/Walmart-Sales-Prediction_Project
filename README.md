# 🛒 Progetto di Previsione delle Vendite Settimanali Walmart

Questo progetto si focalizza sulla modellazione e la previsione delle vendite settimanali (`Weekly_Sales`) dei negozi Walmart. Utilizza tecniche di Machine Learning per analizzare l'influenza di variabili chiave come festività (`IsHoliday`), temperatura, prezzo del carburante e tasso di disoccupazione.

L'analisi esplorativa e i modelli di previsione sono implementati nello script **Prediction_model.py**.

## 🚀 Getting Started

Segui questi passaggi per configurare ed eseguire il progetto localmente.

### Prerequisiti

Assicurati di avere Python (versione 3.x) installato e di utilizzare un ambiente virtuale (es. Conda) per isolare le dipendenze.

### ⚙️ Configurazione dell'Ambiente

1. **Attiva l'ambiente virtuale (es. Conda):**

    ```bash
    conda activate [nome_del_tuo_ambiente]
    ```

2. **Installa le librerie richieste:**

    Il progetto si basa su librerie standard per l'analisi e il Machine Learning (come pandas, numpy e scikit-learn). A seconda del modello specifico utilizzato nel tuo codice, potresti aver bisogno di altre librerie (es. `tensorflow` per l'MLP).

    ```bash
    pip install pandas numpy scikit-learn matplotlib
    # Aggiungi qui altre installazioni specifiche come:
    # pip install tensorflow
    ```

### 💾 Installazione

1. **Clona il repository:**

    ```bash
    git clone [https://github.com/biondomarisol/Walmart-Sales-Prediction-Project.git](https://github.com/biondomarisol/Walmart-Sales-Prediction-Project.git)
    ```

2. **Naviga nella directory del progetto:**

    ```bash
    cd Walmart-Sales-Prediction-Project
    ```

    *Verifica che i file `Prediction_model.py` e `Walmart_Store_sales.csv` siano presenti in questa directory.*

### ▶️ Utilizzo

Esegui lo script principale dalla tua console o terminale:

```bash
python Prediction_model.py
