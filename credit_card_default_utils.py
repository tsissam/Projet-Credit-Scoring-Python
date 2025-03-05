import pandas as pd
import numpy as np
import requests
import zipfile
import os
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

# Fonction pour télécharger les données brutes
def DownloadRawData():
  # URL du fichier ZIP
  url = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"

  # Nom du fichier ZIP (sans chemin, donc dans le répertoire principal)
  zip_filename = "credit_card_data.zip"

  # Téléchargement du fichier ZIP
  response = requests.get(url)
  with open(zip_filename, 'wb') as f:
      f.write(response.content)

  # Extraction du fichier ZIP
  with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
      zip_ref.extractall()

# Fonction pour importer les données brutes
def ReadRawData():
  # Chemin du fichier de données extrait
  data_file = "default of credit card clients.xls"

  # Charger le fichier de données dans un DataFrame Pandas
  data = pd.read_excel(data_file, header=1)  # header=1 pour ignorer la première ligne d'en-tête

  # Returner la dataframe data
  return data


# Reformattage des données brutes
def FormattageRawData():
  # Lire les données brutes
  raw_data = ReadRawData()

  # Renommer les noms de colonne
  raw_data.columns = raw_data.columns.str.lower().str.replace(" ", "_")
  months = ["sep", "aug", "jul", "jun", "may", "apr"]
  variables = ["payment_status", "bill_statement", "previous_payment"]
  new_column_names = [x + "_" + y for x in variables for y in months]
  rename_dict = {x: y for x, y in zip(raw_data.loc[:, "pay_0":"pay_amt6"].columns, new_column_names)}
  raw_data.rename(columns=rename_dict, inplace=True)

  # Mapper les nombres aux chaines de caractères
  gender_dict = {1: "Male",
                2: "Female"}
  education_dict = {0: "Others",
                    1: "Graduate school",
                    2: "University",
                    3: "High school",
                    4: "Others",
                    5: "Others",
                    6: "Others"}
  marital_status_dict = {0: "Others",
                        1: "Married",
                        2: "Single",
                        3: "Others"}
  payment_status = {-2: "Unknown",
                    -1: "Payed duly",
                    0: "Unknown",
                    1: "Payment delayed 1 month",
                    2: "Payment delayed 2 months",
                    3: "Payment delayed 3 months",
                    4: "Payment delayed 4 months",
                    5: "Payment delayed 5 months",
                    6: "Payment delayed 6 months",
                    7: "Payment delayed 7 months",
                    8: "Payment delayed 8 months",
                    9: "Payment delayed >= 9 months"}
  raw_data["sex"] = raw_data["sex"].map(gender_dict)
  raw_data["education"] = raw_data["education"].map(education_dict)
  raw_data["marriage"] = raw_data["marriage"].map(marital_status_dict)

  # Convertir les colonnes 'sex', 'education', 'default_payment_next_month' et 'marriage' en variables catégorielles
  categorical_columns = ['sex', 'marriage', 'education', 'default_payment_next_month']
  raw_data[categorical_columns] = raw_data[categorical_columns].astype('category')

  # Convertir les colonnes payment_status en variables ordinales
  payment_order = list(payment_status.keys())
  payment_categories = pd.CategoricalDtype(categories=payment_order, ordered=True)
  payment_columns = ['payment_status_sep', 'payment_status_aug', 'payment_status_jul',
                     'payment_status_jun', 'payment_status_may', 'payment_status_apr']
  raw_data[payment_columns] = raw_data[payment_columns].astype(payment_categories)

  # Sauvegarde au format csv
  raw_data.to_csv("credit_card_default.csv", index=False)

  # Retourner les données reformattées
  return raw_data


# Fonction pour tracer les distributions de toutes les variables
def plot_distributions(df):
  # Sélectionner uniquement les colonnes numériques
  numeric_columns = df.select_dtypes(include='number').drop(columns=['id'])

  # Afficher un histogramme pour chaque colonne numérique
  plt.figure(figsize=(15, 20))  # Adapter la taille de la figure en fonction du nombre de colonnes
  num_cols = len(numeric_columns.columns)

  # Créer une grille de sous-graphiques adaptée au nombre de colonnes numériques
  for i, column in enumerate(numeric_columns.columns):
      plt.subplot(5, 3, i + 1)  # Adapté à 15 colonnes numériques
      plt.hist(df[column], bins=20, color='blue', alpha=0.7)
      plt.title(f'Histogramme de {column}')
      plt.xlabel(column)
      plt.ylabel('Fréquence')

  plt.tight_layout()
  plt.show()
  plt.close()

  # Afficher des boîtes à moustaches pour chaque colonne numérique (avec axes séparés)
  plt.figure(figsize=(15, 20))  # Adapter la taille de la figure en fonction du nombre de colonnes
  num_cols = len(numeric_columns.columns)

  # Créer une grille de sous-graphiques pour les boîtes à moustaches
  for i, column in enumerate(numeric_columns.columns):
      plt.subplot(5, 3, i + 1)  # Adapté à 15 colonnes numériques
      sns.set(style="whitegrid")
      sns.boxplot(x=df[column], palette="Set2")
      plt.title(f'Boîte à moustaches de {column}')
      plt.xlabel(column)

  plt.tight_layout()
  plt.show()
  plt.close()


  # Colonnes à analyser
  columns_to_analyze = [
      'sex', 'education', 'marriage', 'default_payment_next_month',
      'payment_status_sep', 'payment_status_aug', 'payment_status_jul',
      'payment_status_jun', 'payment_status_may', 'payment_status_apr'
  ]

  # Diagrammes à barres
  for column in columns_to_analyze:
      print(f"Analyse univariée de la colonne '{column}':\n")

      # Compter les occurrences de chaque catégorie
      value_counts = df[column].value_counts(normalize=True)
      print(f"Fréquence des catégories :\n{value_counts}\n")

      # Afficher un graphique à barres pour visualiser la distribution
      plt.figure(figsize=(8, 6))
      sns.countplot(data=df, x=column, palette='Set1')
      plt.title(f'Distribution de {column}')
      plt.xlabel(column)
      plt.ylabel('Fréquence')
      plt.show()
      plt.close()

      # Statistiques descriptives
      print(f"Statistiques descriptives pour {column}:\n")
      print(df[column].describe())

      print("\n" + "="*50 + "\n")


# affichez les distributions en discrétisant suivant une variable catégorielle
def plot_discretize_distributions(df, cat_var):
  # Créer des boîtes à moustaches pour chaque colonne numérique en les segmentant par sexe
  plt.figure(figsize=(15, 20))

  # Sélectionner uniquement les colonnes numériques
  numeric_columns = df.select_dtypes(include='number').drop(columns=['id'])

  # Créer une grille de sous-graphiques pour les boîtes à moustaches
  for i, column in enumerate(numeric_columns.columns):
      plt.subplot(5, 3, i + 1)  # Adapté à 15 colonnes numériques
      sns.set(style="whitegrid")
      sns.boxplot(data=df, x=column, y=cat_var, palette="Set2")
      plt.title(column + ' par ' + cat_var)
      plt.xlabel(column)
      plt.ylabel(cat_var)

  plt.tight_layout()
  plt.show()
  plt.close()


  # Colonnes à analyser par sexe
  columns_to_analyze = [
      'sex', 'education', 'marriage', 'default_payment_next_month',
      'payment_status_sep', 'payment_status_aug', 'payment_status_jul',
      'payment_status_jun', 'payment_status_may', 'payment_status_apr'
  ]
  columns_to_analyze.remove(cat_var)

  # Créer des graphiques à barres pour chaque colonne
  for column in columns_to_analyze:
      plt.figure(figsize=(10, 6))
      sns.countplot(data=df, x=column, hue=cat_var, palette='Set1')

      # Personnalisation du graphique
      plt.title(column + ' par ' + cat_var)
      plt.xlabel(column)
      plt.ylabel('Fréquence')
      plt.xticks(rotation=45)  # Faire pivoter les étiquettes de l'axe des x pour plus de lisibilité

      # Afficher le graphique
      plt.legend(title=cat_var)
      plt.show()
      plt.close()


# Matrice de corrélation
def plot_correlation_matrix(corr_mat):
  sns.set(style="white")
  mask = np.zeros_like(corr_mat, dtype=bool)
  mask[np.triu_indices_from(mask)] = True
  fig, ax = plt.subplots(figsize=(12, 10))
  cmap = sns.diverging_palette(240, 10, n=9, as_cmap=True)
  sns.heatmap(
      corr_mat, mask=mask, cmap=cmap, annot=True,
      fmt=".1f", vmin=-1, vmax=1, center=0, square=True,
      linewidths=.5, cbar_kws={"shrink": .5}, ax=ax
  )
  ax.set_title("Matrice de Correlation", fontsize=16)
  sns.set(style="darkgrid")


def pct_default_by_category(df, cat_var):
  # Pourcentage de défauts de paiement
  ax = df.groupby(cat_var)["default_payment_next_month"] \
  .value_counts(normalize=True) \
  .unstack() \
  .plot(kind="barh", stacked="True")
  ax.set_title("Pourcentage de défauts de paiement",
  fontsize=16)
  ax.legend(title="Defaut de paiement", bbox_to_anchor=(1,1))
  plt.show()


# Fonction d'évaluation des modèles
def performance_evaluation_report(model, X_test, y_test, show_plot=False, labels=None, show_pr_curve=False):
    """
    Function for creating a performance report of a classification model.

    Parameters
    ----------
    model : scikit-learn estimator
        A fitted estimator for classification problems.
    X_test : pd.DataFrame
        DataFrame with features matching y_test
    y_test : array/pd.Series
        Target of a classification problem.
    show_plot : bool
        Flag whether to show the plot
    labels : list
        List with the class names.
    show_pr_curve : bool
        Flag whether to also show the PR-curve. For this to take effect,
        show_plot must be True.

    Return
    ------
    stats : pd.Series
        A series with the most important evaluation metrics
    """

    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    cm = metrics.confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_prob)
    roc_auc = metrics.auc(fpr, tpr)

    precision, recall, _ = metrics.precision_recall_curve(
        y_test, y_pred_prob)
    pr_auc = metrics.auc(recall, precision)

    if show_plot:

        if labels is None:
            labels = ["Negative", "Positive"]

        N_SUBPLOTS = 3 if show_pr_curve else 2
        PLOT_WIDTH = 20 if show_pr_curve else 12
        PLOT_HEIGHT = 5 if show_pr_curve else 6

        fig, ax = plt.subplots(
            1, N_SUBPLOTS, figsize=(PLOT_WIDTH, PLOT_HEIGHT))
        fig.suptitle("Evaluation de la Performance du Modèle", fontsize=16)

        # plot 1: confusion matrix ----

        # preparing more descriptive labels for the confusion matrix
        cm_counts = [f"{val:0.0f}" for val in cm.flatten()]
        cm_percentages = [f"{val:.2%}" for val in cm.flatten()/np.sum(cm)]
        cm_labels = [f"{v1}\n{v2}" for v1, v2 in zip(cm_counts,cm_percentages)]
        cm_labels = np.asarray(cm_labels).reshape(2,2)

        sns.heatmap(cm, annot=cm_labels, fmt="", linewidths=.5, cmap="Greens",
                    square=True, cbar=False, ax=ax[0],
                    annot_kws={"ha": "center", "va": "center"})
        ax[0].set(xlabel="Predicted label",
                  ylabel="Actual label", title="Confusion Matrix")
        ax[0].xaxis.set_ticklabels(labels)
        ax[0].yaxis.set_ticklabels(labels)

        # plot 2: ROC curve ----

        metrics.RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax[1], name="")
        ax[1].set_title("ROC Curve")
        ax[1].plot(fp/(fp+tn), tp/(tp+fn), "ro",
                   markersize=8, label="Decision Point")
        ax[1].plot([0, 1], [0, 1], "r--")

        if show_pr_curve:

            metrics.PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax[2], name="")
            ax[2].set_title("Precision-Recall Curve")

    stats = {
        "accuracy": metrics.accuracy_score(y_test, y_pred),
        "precision": metrics.precision_score(y_test, y_pred),
        "recall": metrics.recall_score(y_test, y_pred),
        "specificity": (tn / (tn + fp)),
        "f1_score": metrics.f1_score(y_test, y_pred),
        "cohens_kappa": metrics.cohen_kappa_score(y_test, y_pred),
        "matthews_corr_coeff": metrics.matthews_corrcoef(y_test, y_pred),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "average_precision": metrics.average_precision_score(y_test, y_pred_prob)
    }

    return stats