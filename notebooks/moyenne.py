import os
from pathlib import Path
import pandas as pd

# 1. Création du nouveau dossier de sortie
dossier_sortie = Path("predictions_combinees")
dossier_sortie.mkdir(parents=True, exist_ok=True)

# 2. Emplacement des fichiers sources
base_dir = Path("soumission") if Path("soumission").exists() else Path(".")
fichiers = [
    base_dir / "reda-soumission_reg_log.csv",
    base_dir / "soumission_knn.csv",
    base_dir / "submission_svm_allvar.csv",
    base_dir / "submission_svm.csv",
]

# 3. Chargement des 4 fichiers
dfs = [pd.read_csv(f) for f in fichiers]

# Identification des colonnes (gère les guillemets éventuels)
id_col = [c for c in dfs[0].columns if c.replace('"', "").upper() == "ID"][0]
target_col = [c for c in dfs[0].columns if c.replace('"', "").upper() == "DIFF"][0]

# 4. Calcul de la moyenne
df_moyenne = pd.DataFrame()
df_moyenne["ID"] = dfs[0][id_col]
df_moyenne["DIFF"] = sum(df[target_col] for df in dfs) / len(dfs)

# 5. Export dans le nouveau dossier
# Version continue (idéale pour maximiser l'AUC sous Kaggle)
chemin_continu = dossier_sortie / "submission_moyenne_continue.csv"
df_moyenne.to_csv(chemin_continu, index=False)

# Version binaire stricte (0 ou 1 avec seuil à 0,5)
df_binaire = df_moyenne.copy()
df_binaire["DIFF"] = (df_binaire["DIFF"] >= 0.5).astype(int)
chemin_binaire = dossier_sortie / "submission_vote_binaire.csv"
df_binaire.to_csv(chemin_binaire, index=False)

print(f"Dossier créé : {dossier_sortie.resolve()}")
print(f"Fichiers enregistrés :")
print(f" - {chemin_continu.name}")
print(f" - {chemin_binaire.name}")
print("\nRépartition des classes (version binaire) :")
print(df_binaire["DIFF"].value_counts())