# ------------------------------------------------------------
# PROGETTO: Previsione delle vendite settimanali Walmart
# MODELLI: Regressione Lineare e Rete Neurale (MLP)
# ------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. CARICAMENTO E PREPARAZIONE DEI DATI
# ------------------------------------------------------------

# Leggo il dataset
data = pd.read_csv("Walmart_Store_sales.csv")
# SALVO UNA COPIA PER I GRAFICI EDA (Time Series)
data_eda = data.copy()

# Converto la data in formato datetime
data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')

# Estraggo informazioni utili dalla data
data['WeekOfYear'] = data['Date'].dt.isocalendar().week
data['Month'] = data['Date'].dt.month
data['DayOfWeek'] = data['Date'].dt.dayofweek

# Trasformo la colonna Store in variabili numeriche (dummy)
data = pd.get_dummies(data, columns=['Store'], drop_first=True)

# Colonne continue da standardizzare
continuous_cols = [
    'Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
    'WeekOfYear', 'Month', 'DayOfWeek'
]

# Tutte le feature usate dai modelli
feature_cols = ['Holiday_Flag'] + continuous_cols + \
               [c for c in data.columns if c.startswith('Store_')]

# Preparo X e y
X = data[feature_cols].copy()
y = np.log1p(data['Weekly_Sales'])     # log per ridurre la variabilità

# Standardizzazione delle colonne continue
scaler = StandardScaler()
X[continuous_cols] = scaler.fit_transform(X[continuous_cols])

# Salvo i nomi delle feature per la predizione futura
final_features = X.columns.tolist()

# Suddivido in train e test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------------
# 2. MODELLI
# ------------------------------------------------------------

# 2a. Regressione Lineare
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

# 2b. Rete Neurale MLP
model_mlp = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42
)
model_mlp.fit(X_train, y_train)

# ------------------------------------------------------------
# 3. VALUTAZIONE DEI MODELLI
# ------------------------------------------------------------

# Ritorno alle scale originali
y_test_real = np.expm1(y_test)
y_train_real = np.expm1(y_train)

# Predizioni sul TEST set
pred_lr_test = np.expm1(model_lr.predict(X_test))
pred_mlp_test = np.expm1(model_mlp.predict(X_test))

# Predizioni sul TRAINING set
pred_lr_train = np.expm1(model_lr.predict(X_train))
pred_mlp_train = np.expm1(model_mlp.predict(X_train))

print("\n--- RISULTATI REGRESSIONE LINEARE ---")
print("R² (Test):", r2_score(y_test_real, pred_lr_test))
print("MSE (Test):", mean_squared_error(y_test_real, pred_lr_test))
print("R² (Train):", r2_score(y_train_real, pred_lr_train))
print("MSE (Train):", mean_squared_error(y_train_real, pred_lr_train))

print("\n--- RISULTATI MLP ---")
print("R² (Test):", r2_score(y_test_real, pred_mlp_test))
print("R² (Train):", r2_score(y_train_real, pred_mlp_train)) # Nuovo
print("MSE (Test):", mean_squared_error(y_test_real, pred_mlp_test))
print("MSE (Train):", mean_squared_error(y_train_real, pred_mlp_train))

# ------------------------------------------------------------
# 4. COEFFICIENTI DELLA REGRESSIONE LINEARE
# ------------------------------------------------------------

coef_df = pd.DataFrame({
    'Feature': final_features,
    'Coefficiente': model_lr.coef_
})

print("\n--- COEFFICIENTI DEL MODELLO LINEARE ---")
print(coef_df.to_string(index=False))
print("\nIntercetta:", model_lr.intercept_)

# ------------------------------------------------------------
# 5. FUNZIONE DI PREDIZIONE PER NUOVI DATI
# ------------------------------------------------------------

def predict_new(input_dict, model_to_use):
    """
    input_dict = dizionario con i valori della nuova osservazione
    model_to_use = modello da usare (model_lr o model_mlp)
    """

    new = pd.DataFrame([input_dict])

    # Creo i dummy per Store
    new = pd.get_dummies(new, columns=['Store'], prefix='Store')

    # Aggiungo eventuali colonne mancanti
    for col in final_features:
        if col not in new.columns:
            new[col] = 0

    # Riordino
    new = new[final_features]

    # Applico lo scaling
    new[continuous_cols] = scaler.transform(new[continuous_cols])

    # Predizione
    pred_log = model_to_use.predict(new)
    return np.expm1(pred_log)[0]

# ------------------------------------------------------------
# 6. ESEMPIO DI PREDIZIONE (INTERATTIVO)
# ------------------------------------------------------------

print("\n--- INSERISCI I VALORI PER LA PREDIZIONE ---")

# Inizializza il dizionario per i nuovi dati
new_example = {}

try:
    # Raccolta dei valori di input dall'utente
    new_example['Store'] = int(input("Numero del Negozio (es. 1, 2, ...): "))
    new_example['Holiday_Flag'] = int(input("È una settimana di festa? (1 per Sì, 0 per No): "))
    new_example['Temperature'] = float(input("Temperatura media (°F): "))
    new_example['Fuel_Price'] = float(input("Prezzo del Carburante (USD/gallone): "))
    new_example['CPI'] = float(input("CPI (Indice Prezzi al Consumo): "))
    new_example['Unemployment'] = float(input("Tasso di Disoccupazione (%): "))
    
    # Per semplicità, chiedo all'utente la data completa e la estraggo in automatico
    date_str = input("Data della settimana (formato gg-mm-aaaa, es. 10-03-2012): ")
    
    # Eseguo l'estrazione delle feature relative alla data
    input_date = pd.to_datetime(date_str, format='%d-%m-%Y')
    new_example['WeekOfYear'] = input_date.isocalendar().week
    new_example['Month'] = input_date.month
    new_example['DayOfWeek'] = input_date.dayofweek # Lunedì=0, Domenica=6
    
    # Converto la WeekOfYear da scalare a intero standard (necessario per lo scaling)
    new_example['WeekOfYear'] = int(new_example['WeekOfYear'])
    
    print("\nElaborazione...")
    
    # Esempio di predizione con i valori inseriti
    print(f"Predizione (Regressione Lineare) per il Negozio {new_example['Store']}: {predict_new(new_example, model_lr):,.2f}")
    print(f"Predizione (MLP) per il Negozio {new_example['Store']}: {predict_new(new_example, model_mlp):,.2f}")

except ValueError:
    print("\nERRORE: Assicurati di inserire numeri per tutti i campi numerici e il formato data corretto.")

# ------------------------------------------------------------
# 7. GRAFICO CONFRONTO PREVISIONI (MODIFICATO E SEPARATO)
# ------------------------------------------------------------

# 1. Trovo il valore minimo e massimo per definire la linea ideale (usiamo un max fisso per coerenza)
min_val = y_test_real.min()
# Definiamo un limite massimo per l'asse X e Y (Zoom) per migliorare la leggibilità
# Lo imposto a circa il valore massimo ma leggermente arrotondato
# o puoi usare un valore fisso come 3.0e6 per lo zoom sui dati comuni
max_display_val = max(y_test_real.max(), pred_lr_test.max(), pred_mlp_test.max())
# Usiamo 2.5 milioni come limite di zoom per concentrarci sulla maggior parte dei dati
ZOOM_LIMIT = 2.5e6

plt.figure(figsize=(14, 6))

# --- SUBPLOT 1: Regressione Lineare (LR) ---
plt.subplot(1, 2, 1)

# Disegno la linea ideale (y = x)
plt.plot([min_val, max_display_val], [min_val, max_display_val], color='red', linestyle='--', label='Predizione Ideale')

# Scatter plot per LR
plt.scatter(y_test_real, pred_lr_test, label='Linear Regression', alpha=0.7, color='skyblue')

# Applichiamo lo zoom per la leggibilità
plt.xlim(0, ZOOM_LIMIT)
plt.ylim(0, ZOOM_LIMIT)

plt.xlabel("Valori Reali ($)")
plt.ylabel("Valori Predetti ($)")
plt.title("Accuratezza Predizioni: Regressione Lineare")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)


# --- SUBPLOT 2: Rete Neurale (MLP) ---
plt.subplot(1, 2, 2)

# Disegno la linea ideale (y = x)
plt.plot([min_val, max_display_val], [min_val, max_display_val], color='red', linestyle='--', label='Predizione Ideale')

# Scatter plot per MLP
plt.scatter(y_test_real, pred_mlp_test, label='MLP', alpha=0.7, color='green')

# Applichiamo lo zoom per la leggibilità
plt.xlim(0, ZOOM_LIMIT)
plt.ylim(0, ZOOM_LIMIT)

plt.xlabel("Valori Reali ($)")
plt.ylabel("Valori Predetti ($)")
plt.title("Accuratezza Predizioni: Rete Neurale (MLP)")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()








plt.figure(figsize=(12, 5))
# Residui per Regressione Lineare
residuals_lr = y_test_real - pred_lr_test
plt.subplot(1, 2, 1)
plt.scatter(pred_lr_test, residuals_lr, alpha=0.5)
plt.hlines(0, pred_lr_test.min(), pred_lr_test.max(), color='red', linestyle='--')
plt.title('Residui vs Predizioni (Reg. Lineare)')
plt.xlabel('Valori Predetti')
plt.ylabel('Residui')
# Residui per MLP
residuals_mlp = y_test_real - pred_mlp_test
plt.subplot(1, 2, 2)
plt.scatter(pred_mlp_test, residuals_mlp, alpha=0.5, color='green')
plt.hlines(0, pred_mlp_test.min(), pred_mlp_test.max(), color='red', linestyle='--')
plt.title('Residui vs Predizioni (MLP)')
plt.xlabel('Valori Predetti')
plt.ylabel('Residui')
plt.tight_layout()
plt.show()

# GRAFICO 1 RIVISTO: Confronto Vendite Medie per Holiday_Flag
# Raggruppo le vendite medie per la bandiera festiva
# NOTA: Assumo che 'data_eda' e 'data' siano ancora caricate come nel codice originale
holiday_comparison = data_eda.groupby('Holiday_Flag')['Weekly_Sales'].mean()

plt.figure(figsize=(8, 6))
# Rinomino le etichette per chiarezza
labels = ['Settimana Normale (0)', 'Settimana Festiva (1)']
bars = plt.bar(labels, holiday_comparison.values, color=['skyblue', 'lightcoral'])

# NUOVA POSIZIONE DEI VALORI: In basso all'interno della barra
for bar in bars:
    yval = bar.get_height()
    # Posiziona il testo 100,000 $ sopra l'asse x per essere visibile all'interno
    plt.text(bar.get_x() + bar.get_width()/2, 100000,
             f'${yval:,.0f}',
             ha='center', va='bottom', fontsize=12, color='black', weight='bold')

plt.title('Vendite Settimanali Medie: Festività vs. Settimane Normali')
plt.xlabel('Holiday_Flag')
plt.ylabel('Vendite Settimanali Medie ($)')
# Imposto il limite dell'asse y per avere più spazio sopra le barre
plt.ylim(0, holiday_comparison.max() * 1.05)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# GRAFICO 2 RIVISTO: Andamento Stagionale (Mesi e Festività)

# Mappa per convertire il numero della settimana in nome del mese approssimativo (per etichette)
# Mesi: Gen (1-4), Feb (5-8), Mar (9-13), Apr (14-17), Mag (18-22), Giu (23-26),
# Lug (27-30), Ago (31-35), Set (36-39), Ott (40-44), Nov (45-48), Dic (49-52)
month_mapping = {
    4: 'Gen', 8: 'Feb', 13: 'Mar', 17: 'Apr', 22: 'Mag', 26: 'Giu',
    30: 'Lug', 35: 'Ago', 39: 'Set', 44: 'Ott', 48: 'Nov', 52: 'Dic'
}

# 1. Raggruppo le vendite medie per la Settimana dell'Anno
seasonal_sales = data.groupby('WeekOfYear')['Weekly_Sales'].mean().reset_index()

plt.figure(figsize=(15, 6))
plt.plot(seasonal_sales['WeekOfYear'], seasonal_sales['Weekly_Sales'],
         marker='o', linestyle='-', color='darkgreen')

# 2. Imposto le etichette dell'asse X con i Mesi
week_ticks = list(month_mapping.keys())
month_labels = list(month_mapping.values())
plt.xticks(week_ticks, month_labels, rotation=0)

plt.title('Andamento Stagionale delle Vendite (Vendite Medie per Settimana)')
plt.xlabel('Periodo dell\'Anno')
plt.ylabel('Vendite Settimanali Medie ($)')
plt.grid(True, linestyle='--', alpha=0.7)

# 3. Evidenziare i picchi con NOMI DI FESTIVITÀ per una comprensione immediata
# NOTA: Queste sono le settimane standard per le maggiori festività in USA nel periodo (WalMart data)
peak_annotations = {
    # Circa 47: Ringraziamento (Thanksgiving)
    47: {'name': 'Thanksgiving', 'color': 'red'},
    # Circa 50: Settimana prima di Natale (Black Friday/Anticipo Natalizio)
    50: {'name': 'Black Friday', 'color': 'red'},
    # Circa 51: Natale/Capodanno
    51: {'name': 'Sett. di Natale', 'color': 'red'}
}

for week, info in peak_annotations.items():
    # Trova il valore delle vendite per quella settimana
    peak_value = seasonal_sales[seasonal_sales['WeekOfYear'] == week]['Weekly_Sales'].iloc[0]

    plt.scatter(week, peak_value, color=info['color'], s=150, zorder=5)
    plt.annotate(
        info['name'],
        (week, peak_value),
        textcoords="offset points",
        xytext=(0, 15), # Sposta l'etichetta in alto
        ha='center',
        color=info['color'],
        weight='bold',
        fontsize=10
    )

plt.show()
