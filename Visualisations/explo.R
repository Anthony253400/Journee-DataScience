## ============================================================
## GRAPHIQUES - Analyse exploratoire (sans FactoMineR/factoextra)
## Exploitations agricoles (DIFF = saine / défaillante)
## Dépendances : uniquement ggplot2 (+ base R)
## ============================================================

# install.packages("ggplot2")
library(ggplot2)

## --- Chargement ---
data <- read.csv("C:/Users/bruno/Journee-DataScience/data/farms_train.csv", header = TRUE, sep = ",")
data$DIFF <- as.factor(data$DIFF)
vars_num <- setdiff(colnames(data), "DIFF")


## ============================================================
## 1. Répartition des classes (équilibre du jeu de données)
## ============================================================
ggplot(data, aes(x = DIFF, fill = DIFF)) +
  geom_bar() +
  geom_text(stat = "count", aes(label = after_stat(count)), vjust = -0.5) +
  labs(title = "Répartition des exploitations saines / défaillantes",
       x = "DIFF (0 = défaillante, 1 = saine)", y = "Nombre d'exploitations") +
  theme_minimal()
# A regarder : classes équilibrées ou pas ? Si très déséquilibré,
# ça justifie de privilégier l'AUC plutôt que l'accuracy.


## ============================================================
## 2. Distributions de chaque variable, par classe (boxplots)
## ============================================================
# On passe les données au format "long" avec la fonction de base reshape()
data_long <- reshape(
  data,
  varying = vars_num,
  v.names = "value",
  timevar = "variable",
  times = vars_num,
  direction = "long"
)

ggplot(data_long, aes(x = DIFF, y = value, fill = DIFF)) +
  geom_boxplot() +
  facet_wrap(~ variable, scales = "free_y") +
  labs(title = "Distribution de chaque variable selon la classe DIFF") +
  theme_minimal() +
  theme(legend.position = "none")
# A regarder : les variables où les boîtes des deux classes sont bien
# séparées sont les plus discriminantes (bons candidats pour le modèle).


## ============================================================
## 3. Densités superposées (autre vue que le boxplot)
## ============================================================
ggplot(data_long, aes(x = value, fill = DIFF)) +
  geom_density(alpha = 0.5) +
  facet_wrap(~ variable, scales = "free") +
  labs(title = "Densité de chaque variable selon la classe DIFF") +
  theme_minimal()
# A regarder : deux courbes bien décalées horizontalement = variable
# discriminante. Deux courbes qui se chevauchent presque = variable peu utile seule.


## ============================================================
## 4. Pairplot (croisement de toutes les variables) - base R
## ============================================================
couleurs <- c("firebrick", "steelblue")[data$DIFF]
pairs(data[, vars_num],
      col = couleurs,
      pch = 19,
      main = "Croisement des variables (rouge = 0, bleu = 1)")
# A regarder : les nuages de points où rouge et bleu forment deux
# groupes visuellement séparés.


## ============================================================
## 5. Matrice de corrélation - heatmap avec ggplot2 (sans corrplot)
## ============================================================
R <- cor(data[, vars_num])
R_melt <- as.data.frame(as.table(R))
colnames(R_melt) <- c("Var1", "Var2", "Correlation")

ggplot(R_melt, aes(x = Var1, y = Var2, fill = Correlation)) +
  geom_tile(color = "white") +
  geom_text(aes(label = round(Correlation, 2)), size = 3) +
  scale_fill_gradient2(low = "firebrick", mid = "white", high = "steelblue",
                       midpoint = 0, limits = c(-1, 1)) +
  labs(title = "Matrice de corrélation", x = "", y = "") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
# A regarder : coefficients proches de 1 ou -1 = variables redondantes
# (multicolinéarité), à signaler surtout pour la régression logistique.


## ============================================================
## 6. ACP avec prcomp() - fonction native de R, aucune librairie
## ============================================================
pca <- prcomp(data[, vars_num], scale. = TRUE)  # scale. = TRUE = standardisation

# 6a. Scree plot : variance expliquée par composante
var_explained <- (pca$sdev^2) / sum(pca$sdev^2) * 100
scree_df <- data.frame(Composante = factor(1:length(var_explained)),
                       Variance = var_explained)

ggplot(scree_df, aes(x = Composante, y = Variance)) +
  geom_col(fill = "steelblue") +
  geom_text(aes(label = paste0(round(Variance, 1), "%")), vjust = -0.5) +
  labs(title = "Variance expliquée par composante", y = "% de variance") +
  theme_minimal()

# 6b. Cercle des corrélations (biplot des variables) - base R
biplot(pca, cex = 0.7, main = "Cercle des corrélations (ACP)")
# A regarder : la direction et la longueur des flèches (variables) -
# flèches proches = variables corrélées entre elles.

# 6c. Projection des individus, colorée par la vraie classe
scores <- as.data.frame(pca$x[, 1:2])  # coordonnées sur les 2 premières composantes
scores$DIFF <- data$DIFF

ggplot(scores, aes(x = PC1, y = PC2, color = DIFF)) +
  geom_point(alpha = 0.7) +
  stat_ellipse(level = 0.68) +
  scale_color_manual(values = c("firebrick", "steelblue")) +
  labs(title = "Projection des individus sur les 2 premières composantes",
       x = paste0("PC1 (", round(var_explained[1], 1), "%)"),
       y = paste0("PC2 (", round(var_explained[2], 1), "%)")) +
  theme_minimal()
# A regarder : si les ellipses des deux classes se recouvrent beaucoup,
# la séparation n'est pas triviale en 2D -> les modèles auront du travail.


## ============================================================
## 7. Corrélation de chaque variable avec la cible (bar chart)
## ============================================================
data_num <- data
data_num$DIFF <- as.numeric(as.character(data_num$DIFF))
cor_target <- sapply(vars_num, function(v) cor(data_num[[v]], data_num$DIFF))
cor_df <- data.frame(variable = names(cor_target), correlation = cor_target)

ggplot(cor_df, aes(x = reorder(variable, correlation), y = correlation, fill = correlation > 0)) +
  geom_col() +
  coord_flip() +
  labs(title = "Corrélation de chaque variable avec DIFF",
       x = "", y = "Corrélation") +
  theme_minimal() +
  theme(legend.position = "none")
# A regarder : les barres les plus longues (positives ou négatives) sont
# les variables les plus prometteuses individuellement pour prédire DIFF.

## ============================================================
## FIN
## ============================================================
