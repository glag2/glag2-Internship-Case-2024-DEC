import pandas as pd
from sklearn.ensemble import IsolationForest
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
sklearn.set_config(transform_output="pandas")
from tabulate import tabulate
from termcolor import colored
from statsmodels.graphics.mosaicplot import mosaic
from sklearn.preprocessing import PowerTransformer
from scipy import stats
from statsmodels.graphics.mosaicplot import mosaic
from scipy.stats import chi2_contingency
import matplotlib.patches as mpatches
from scipy.stats import yeojohnson
from scipy.stats import gaussian_kde

def show_residuals_plot(y_pred, y_test):
    """
    Mostra un grafico a dispersione dei valori predetti rispetto ai valori effettivi,
    con una mappa di densità dei punti.

    :param y_pred: Valori predetti dal modello.
    :param y_test: Valori effettivi.
    """
    # Calcola la densità dei punti
    density = gaussian_kde(np.stack([y_pred, y_test]))(np.stack([y_pred, y_test]))

    # Crea il grafico
    plt.figure(figsize=(15, 7))
    plt.scatter(y_pred, y_test, s=5, alpha=0.3, linewidths=0.5, c=density, cmap="rocket")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Predicted vs Actual Residuals")
    plt.colorbar(label="Density")
    plt.show()

def hexbin_plot(df, x_col, y_col, gridsize=50, cmap='gnuplot', figsize=(10, 8)):
    """
    Crea un grafico esagonale (hexbin) per visualizzare la densità di punti
    in un grafico scatter bidimensionale.

    :param df: DataFrame contenente i dati.
    :param x_col: Nome della colonna sull'asse x.
    :param y_col: Nome della colonna sull'asse y.
    :param gridsize: Dimensione della griglia esagonale.
    :param cmap: Mappa di colori da utilizzare.
    :param figsize: Dimensioni del grafico (larghezza, altezza).
    """
    plt.figure(figsize=figsize)
    plt.hexbin(df[x_col], df[y_col], gridsize=gridsize, cmap=cmap, mincnt=1)
    plt.colorbar(label='Count')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title('Hexbin plot')
    plt.show()

def multi_variable_scatter_plot(df, num_col1, num_col2, cat_col, figsize=(8,8)):
    """
    Crea un grafico scatter con istogrammi marginali dati un DataFrame e
    i nomi di due colonne numeriche e una colonna categorica.

    :param df: DataFrame contenente i dati.
    :param num_col1: Nome della prima colonna numerica (es. 'Age').
    :param num_col2: Nome della seconda colonna numerica (es. 'Fare').
    :param cat_col: Nome della colonna categorica (es. 'Sex').
    :param figsize: Dimensioni del grafico (larghezza, altezza).
    """
    sns.set(style="whitegrid")

    # Crea il plot
    g = sns.jointplot(
        data=df, 
        x=num_col1, 
        y=num_col2, 
        hue=cat_col, 
        kind="scatter", 
        marginal_kws=dict(fill=True),
        height=figsize[0],  # Usare la larghezza per l'altezza del jointplot
        alpha=0.6
    )

    # aggiungo delle linee di separazione in base alla colonna categorica
    for cat in df[cat_col].unique():
        # Estrai i dati relativi alla categoria e rimuovi righe con NaN
        cat_data = df[df[cat_col] == cat][[num_col1, num_col2]].dropna()

        x = cat_data[num_col1]
        y = cat_data[num_col2]

        # Controllo se i dati sono sufficienti per la regressione
        if len(x) > 1 and x.nunique() > 1 and y.nunique() > 1:
            # Calcola la retta di regressione
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            line = slope * x + intercept

            # Disegna la retta
            plt.plot(x, line, label=f"{cat} - $R^2$={r_value**2:.2f}")

            # stampo i dettagli della regressione
            print(f"Regressione per {cat}:")
            print(f"R^2: {r_value**2:.2f}")
        else:
            print(f"Campioni insufficienti o varianza zero per {cat}, nessuna regressione possibile.")

    # Aggiungi titolo
    plt.suptitle(f"{num_col1} vs {num_col2} Scatterplot by {cat_col}", y=1.03)

    # Mostra il grafico
    plt.show()

def raincloud_plot(data, numerical_col, categorical_col, group_col, point_color='lightcoral', figsize=(15, 10), yeo_johnson=False):
    """
    Genera un Raincloud Plot per confrontare una variabile numerica rispetto a una variabile categoriale,
    suddivisa per un'altra variabile di raggruppamento.

    Args:
        data (pd.DataFrame): Il dataframe contenente i dati.
        numerical_col (str): Il nome della colonna che contiene i valori numerici.
        categorical_col (str): Il nome della colonna che contiene i valori categorici.
        group_col (str): Il nome della colonna che contiene i gruppi di suddivisione.
        point_color (str): Colore uniforme per i puntini del grafico.
        figsize (tuple): Dimensioni della figura.
        yeo_johnson (bool): Se True, applica la trasformazione Yeo-Johnson alla colonna numerica.
    
    Returns:
        None. Visualizza il grafico.
    """

    data = data.copy()
    
    # Converti la colonna categoriale in tipo stringa per evitare conflitti
    data[categorical_col] = data[categorical_col].astype(str)
    data[group_col] = data[group_col].astype(str)  # Assicura che anche il group_col sia in formato stringa

    # Applica la trasformazione Yeo-Johnson se richiesto
    if yeo_johnson:
        pt_transformer = PowerTransformer(method='yeo-johnson')
        data[numerical_col] = pt_transformer.fit_transform(data[[numerical_col]])

    # Rimuovi eventuali valori nulli
    data = data.dropna(subset=[numerical_col, categorical_col, group_col])

    # Crea il grafico
    fig, ax = plt.subplots(figsize=figsize)
    
    # 1. Aggiungi un boxplot unico per ogni categoria
    sns.boxplot(x=categorical_col, y=numerical_col, data=data, 
                color='lightgray', dodge=False, ax=ax, fliersize=0, linewidth=1, zorder=1)

    # 2. Grafico a dispersione con colore uniforme
    sns.stripplot(x=categorical_col, y=numerical_col, data=data, hue=group_col,
                    palette="viridis", alpha=0.3, ax=ax, dodge=True, jitter=True ,legend=False, zorder=2)

    # 3. Grafico a violino (sopra agli altri)
    sns.violinplot(x=categorical_col, y=numerical_col, data=data, 
                   hue=group_col, palette="autumn", dodge=True, width=0.8, linewidth=0, zorder=3, alpha=0.8)
    
    # Imposta i titoli e le etichette
    title = f"{('YeoJohnson (' + numerical_col + ')') if yeo_johnson else numerical_col} vs {categorical_col} Raincloud Plot by {group_col}"
    ax.set_title(title, size=18)

    ax.set_xlabel(categorical_col, size=14)
    ax.set_ylabel(numerical_col if not yeo_johnson else f'YeoJohnson({numerical_col})', size=14)

    # Mostra la legenda a destra
    ax.legend(title=group_col, loc='upper right')

    plt.tight_layout()
    plt.show()

def create_mosaic_plot_with_legend(df, columns, figsize=(10, 6)):
    """
    Create a mosaic plot based on the specified columns of a DataFrame, with color coding 
    based on standardized residuals from the chi-square test, and includes a legend.

    Parameters:
    - df: The pandas DataFrame containing the data.
    - columns: List of column names to include in the mosaic plot.
    - figsize: Tuple specifying the size of the figure (width, height).

    Example usage:
    create_mosaic_plot_with_legend(data, ['col1', 'col2'], figsize=(12, 8))
    """
    
    # Ensure the columns provided exist in the DataFrame
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
    
    # Convert boolean columns to string (True -> 'True', False -> 'False')
    df[columns[0]] = df[columns[0]].astype(str)
    df[columns[1]] = df[columns[1]].astype(str)
    
    # Creating a contingency table with the provided columns
    contingency_table = pd.crosstab(df[columns[0]], df[columns[1]])
    
    # Chi-square test to calculate residuals
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    residuals = (contingency_table - expected) / (expected ** 0.5)
    
    # Defining a function to color based on residuals
    def color_by_residual(key):
        row, col = key  # Extract row and column labels
        observed_value = contingency_table.loc[row, col]
        expected_value = expected[contingency_table.index.get_loc(row), contingency_table.columns.get_loc(col)]
        residual = (observed_value - expected_value) / (expected_value ** 0.5)
        
        if residual > 2:
            return '#ff9999'  # Strong positive residual (red)
        elif residual < -2:
            return '#9999ff'  # Strong negative residual (blue)
        else:
            return '#cccccc'  # Neutral residual (gray)
    
    # Plotting the mosaic plot with colors based on residuals
    fig, _ = mosaic(contingency_table.stack(), gap=0.02, properties=lambda key: {'color': color_by_residual(key)})
    # aggiungo agli assi il nome delle colonne
    fig.axes[0].set_xlabel(columns[0])
    fig.axes[0].set_ylabel(columns[1])
    plt.title(f'Mosaic Plot with Residuals: {columns[0]} vs {columns[1]}')
    fig.set_size_inches(figsize)

    # Create custom legend patches
    legend_elements = [
        mpatches.Patch(color='#ff9999', label='Residual > 2 (Positive)'),
        mpatches.Patch(color='#9999ff', label='Residual < -2 (Negative)'),
        mpatches.Patch(color='#cccccc', label='Residual ~ 0 (Neutral)')
    ]
    
    # Add the custom legend to the plot
    plt.legend(handles=legend_elements, title='Standardized Residuals', loc='upper right')
    
    plt.show()
    # Printing residuals for reference
    print("Chi-square Test Residuals:")
    print(residuals)

def apply_yeojohnson_manual(data, column):
    """
    Applies the Yeo-Johnson transformation manually using scipy.stats.
    
    Parameters:
        data (pd.DataFrame): The input DataFrame.
        column (str): The column to transform.

    Returns:
        pd.Series: The transformed column.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in the DataFrame.")
    
    # Remove NaNs before transformation
    data_clean = data[column].dropna()

    # Apply Yeo-Johnson transformation
    transformed, lambda_ = yeojohnson(data_clean)
    print(f"Optimal lambda for {column}: {lambda_}")

    # Replace original values in the column
    data.loc[data_clean.index, column] = transformed
    return data[column]

def plot_numeric_column(data, column, color, contrast, exclude_outliers=True, bins=50, figsize=(25, 10), yeojohnson = False):
    data = data.copy()
    if yeojohnson:
        data[column] = apply_yeojohnson_manual(data, column)

    # Calcola i quartili e l'IQR
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1

    # Definisce i limiti per identificare gli outlier
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Separazione degli outliers
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

    # Ordina gli outlier in base alla colonna specificata
    outliers_sorted = outliers.sort_values(by=column)

    # Stampa gli outliers utilizzando tabulate se presenti
    if not outliers_sorted.empty:
        print(f"\nOutliers per la variabile {column}:\n")

        # Evidenzia la colonna degli outliers
        def highlight_column(row, highlight_col):
            return [colored(cell, 'red') if idx == highlight_col else cell for idx, cell in enumerate(row)]

        headers = outliers_sorted.columns.tolist()
        table_data = [highlight_column(row, headers.index(column)) for row in outliers_sorted.values]

        # Stampa la tabella evidenziata
        print(tabulate(table_data, headers=headers, tablefmt='grid'))

    # Scegli i dati da utilizzare in base a exclude_outliers
    data_to_plot = filtered_data if exclude_outliers else data

    # Creazione del grafico con o senza outlier in base a exclude_outliers
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})

    # Istogramma
    sns.histplot(data_to_plot[column], kde=False, ax=ax1, bins=bins, color=color, stat='count')
    title_suffix = "(Senza Outliers)" if exclude_outliers else "(Con Outliers)"
    ax1.set_title(f'Analisi Univariata della Variabile {column} {title_suffix}')
    ax1.set_xlabel(column)
    ax1.set_ylabel('Conteggio')

    # Boxplot e Stripplot (con KDE)
    min_xlim, max_xlim = ax1.get_xlim()
    sns.boxplot(x=data_to_plot[column], ax=ax2, color=color)
    sns.kdeplot(data_to_plot[column], ax=ax2, color=color, fill=True, linewidth=2)
    sns.stripplot(x=data_to_plot[column], ax=ax2, color=contrast, alpha=0.5, jitter=True)

    ax2.set_xlim(min_xlim, max_xlim)
    ax2.set_xlabel(column)

    plt.tight_layout()
    plt.show()


def detect_anomalous_strings(dict, contamination=0.1):
    """
    Funzione che individua stringhe anomale in un dizionario usando Isolation Forest.
    
    Parametri:
    - dict: Dizionario contenente le stringhe da analizzare.
    - contamination (float): Percentuale di valori considerati come outlier (default 0.1).
    
    Ritorna:
    - anomalous_strings (list): Lista di stringhe anomale.
    """

    # Funzione per calcolare le feature (qui solo la lunghezza della stringa)
    def calculate_features(strings):
        features = []
        for s in strings:
            features.append([len(s)])  # Usa la lunghezza della stringa come feature
        return np.array(features)

    # Crea un elenco di tutte le stringhe
    all_strings = []
    string_to_freq_map = {}  # Per tracciare la frequenza associata a ogni stringa
    for freq, values in dict.items():
        for value in values:
            all_strings.append(value)
            string_to_freq_map[value] = freq

    # Calcola le feature per ogni stringa
    features = calculate_features(all_strings)

    # Applica Isolation Forest per trovare outlier
    model = IsolationForest(contamination=contamination)  # Controlla il livello di outlier desiderato
    outlier_labels = model.fit_predict(features)

    # Raccoglie le stringhe anomale
    anomalous_strings = []
    for i, string in enumerate(all_strings):
        if outlier_labels[i] == -1:  # -1 indica un outlier
            anomalous_strings.append((string, string_to_freq_map[string]))

    # Stampa le stringhe anomale con la loro frequenza
    if anomalous_strings:
        print("\n\nStringhe anomale rilevate:")
        print(tabulate(anomalous_strings, headers=["Stringa", "Frequenza"], tablefmt='grid'))
    else:
        print("\n\nNessuna stringa anomala rilevata.")

    return anomalous_strings

def plot_categorical_column(data, column, color, min_percentile=0, max_percentile=100, hide_outliers=False, figsize=(25, 10), anomaly_threshold=0.006):
    # Step 1: Prepara i dati
    data = data.copy()
    data[column] = data[column].astype('object').fillna('Missing')

    # Step 2: Calcola i conteggi per ciascuna categoria
    value_counts = data[column].value_counts()

    # Step 3: Crea un'altra copia per il primo grafico
    data_for_first_plot = data.copy()

    # Step 4: Sostituisci i valori nel dataframe con la stringa "Frequenza n"
    def replace_with_frequency(val):
        freq = value_counts[val] if val in value_counts.index else None
        return f'Frequenza {freq}' if freq is not None else val

    data_for_first_plot[column] = data_for_first_plot[column].apply(replace_with_frequency)

    # Step 5: Conta quante volte appare ciascuna frequenza
    frequency_of_frequencies = data_for_first_plot[column].value_counts()

    # Step 6: Ordina le categorie in base alla frequenza (Frequenza 1, Frequenza 2, etc.)
    sorted_categories = sorted(frequency_of_frequencies.index, key=lambda x: int(x.split()[1]))

    # Step 7: Correggi i valori dividendo per la frequenza
    corrected_counts = []
    for category in sorted_categories:
        freq_num = int(category.split()[1])  # Ottieni il numero di frequenza
        corrected_count = frequency_of_frequencies[category] // freq_num  # Dividi il numero di occorrenze per la frequenza
        corrected_counts.append(corrected_count)

    # Step 8: Stampa le categorie e le loro frequenze
    all_values_by_frequency = {}
    for freq in sorted(value_counts.unique()):
        values = sorted(value_counts[value_counts == freq].index.tolist())
        all_values_by_frequency[f'Frequenza {freq}'] = values
    
    print("\nCategorie per Frequenza:")
    for label, values in all_values_by_frequency.items():
        print(f"{label}: {', '.join(values)}")

    # stampo i valori anomali
    detect_anomalous_strings(all_values_by_frequency, contamination=anomaly_threshold)

    # Step 9: Grafico della distribuzione delle frequenze corrette (Verticale)
    plt.figure(figsize=figsize)
    sns.set_style('whitegrid')

    # Grafico per la distribuzione delle frequenze con le categorie sostituite e ordinate
    sns.barplot(x=sorted_categories, y=corrected_counts, color=color)
    plt.title(f'Distribuzione del numero di occorrenze per ciascuna Frequenza ({column})')
    plt.xticks(rotation=75)
    plt.xlabel('Frequenza')
    plt.ylabel('Numero di Occorrenze')

    # Aggiungi annotazioni con i conteggi sopra le barre
    for p, count in zip(plt.gca().patches, corrected_counts):
        height = p.get_height()
        plt.gca().annotate(f'{int(count)}', (p.get_x() + p.get_width() / 2, height + 0.5),
                           ha='center', va='bottom')

    # Step 10: Calcola i percentili di inclusione per il secondo grafico
    min_threshold = value_counts.quantile(min_percentile / 100)
    max_threshold = value_counts.quantile(max_percentile / 100)

    # Assicurati che i percentili siano ordinati correttamente
    if min_threshold > max_threshold:
        min_threshold, max_threshold = max_threshold, min_threshold

    print(f"\nIl valore del percentile minimo ({min_percentile}%) è: {min_threshold}")
    print(f"Il valore del percentile massimo ({max_percentile}%) è: {max_threshold}")

    # Step 11: Identifica le categorie da sostituire e quelle da mantenere
    keep_values = value_counts[(value_counts >= min_threshold) & (value_counts <= max_threshold)]
    replace_values = value_counts[~value_counts.index.isin(keep_values.index)]

    # Stampa le frequenze escluse e mantenute
    print(f"\nFrequenze escluse: {', '.join(map(str, replace_values.unique()))}")
    print(f"Frequenze mantenute: {', '.join(map(str, keep_values.unique()))}")

    # Step 12: Sostituisci i valori nel dataframe per il secondo grafico
    def replace_outliers(val):
        freq = value_counts[val] if val in value_counts.index else None
        return f'Frequenza {freq}' if (freq is not None and (freq < min_threshold or freq > max_threshold)) else val

    # Crea una copia del dataframe per il grafico finale
    data_for_final_plot = data.copy()

    if hide_outliers:
        # Se hide_outliers è True, rimuovi i valori esclusi
        data_for_final_plot = data_for_final_plot[data_for_final_plot[column].isin(keep_values.index)]
    else:
        # Altrimenti, sostituisci i valori esclusi
        data_for_final_plot[column] = data_for_final_plot[column].apply(replace_outliers)

    # Step 13: Correggi i valori sostituiti dividendo per la frequenza
    def divide_corrected_frequencies(val):
        if 'Frequenza' in val:
            freq_num = int(val.split()[1])  # Ottieni il numero di frequenza
            return val, freq_num
        return val, 1

    frequency_counts = data_for_final_plot[column].value_counts()
    corrected_frequencies_final = {}
    for category, count in frequency_counts.items():
        category_name, freq_num = divide_corrected_frequencies(category)
        corrected_frequencies_final[category_name] = count // freq_num

    # Ordina le categorie corrette
    sorted_corrected_categories = sorted(corrected_frequencies_final.keys(), key=lambda x: int(x.split()[1]) if 'Frequenza' in x else float('inf'))
    corrected_final_counts = [corrected_frequencies_final[cat] for cat in sorted_corrected_categories]

    # Step 14: Grafico finale con frequenze mantenute e sostituite (Verticale)
    plt.figure(figsize=figsize)
    sns.set_style('whitegrid')

    # Grafico per la distribuzione con le categorie mantenute e quelle escluse
    sns.barplot(x=sorted_corrected_categories, y=corrected_final_counts, color=color)
    plt.title(f'Distribuzione della Variabile Categorica {column} (con o senza outliers)')
    plt.xticks(rotation=75)
    # sposto il titolo a sinistra per allinearlo con la colonna
    plt.xlabel(column)
    plt.ylabel('Conteggio')

    # Aggiungi annotazioni con i conteggi sopra le barre
    for p in plt.gca().patches:
        height = p.get_height()
        plt.gca().annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2, height + 0.5),
                           ha='center', va='bottom')

    plt.show()