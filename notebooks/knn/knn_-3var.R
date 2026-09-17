## ============================================================
## KNN - Choix du k, courbe ROC, AUC et visualisation PCA
## ============================================================

# install.packages(c("class", "pROC"))
library(class)   # fonction knn()
library(pROC)    # calcul ROC / AUC

## --- 1. Chargement des données ---
data <- read.csv(
  "C:/Users/bruno/Journee-DataScience/data/farms_train.csv",
  header = TRUE,
  sep = ","
)

data$DIFF <- as.factor(data$DIFF)


## ============================================================
## 2. Split train / validation
## ============================================================
## On sépare les données en :
## - 80% pour entraîner le KNN
## - 20% pour choisir k et évaluer le modèle
##
## set.seed(42) permet d'obtenir toujours le même split.

set.seed(42)

n <- nrow(data)

idx <- sample(1:n, size = 0.8 * n)

train_set <- data[idx, ]
valid_set <- data[-idx, ]


## ============================================================
## 3. Sélection des variables utilisées par le KNN
## ============================================================
## On retire :
## - DIFF : variable cible, donc elle ne doit pas servir
##   à calculer les distances
## - AGE
## - R22
## - R8
##
## Le KNN utilisera toutes les autres variables numériques.

vars_num <- setdiff(
  colnames(data),
  c("DIFF", "AGE", "R22", "R8")
)


## ============================================================
## 4. Standardisation des variables
## ============================================================
## KNN se base sur des distances.
##
## Si les variables n'ont pas la même échelle,
## les variables ayant de grandes valeurs peuvent
## dominer le calcul de distance.
##
## On calcule donc la moyenne et l'écart-type
## UNIQUEMENT sur le train.
##
## Ensuite, on utilise ces mêmes valeurs pour
## standardiser le validation set.
##
## Cela évite la fuite de données.

train_means <- sapply(train_set[, vars_num], mean)
train_sds   <- sapply(train_set[, vars_num], sd)


## Fonction permettant de standardiser les données

standardize <- function(df, means, sds) {
  as.data.frame(
    scale(df, center = means, scale = sds)
  )
}


## Standardisation du train

x_train <- standardize(
  train_set[, vars_num],
  train_means,
  train_sds
)


## Standardisation du validation

x_valid <- standardize(
  valid_set[, vars_num],
  train_means,
  train_sds
)


## Variables cibles

y_train <- train_set$DIFF
y_valid <- valid_set$DIFF


## ============================================================
## 5. Choix du meilleur k via l'AUC
## ============================================================
## On teste plusieurs valeurs de k.
##
## Pour chaque k :
## 1. Le KNN classe les observations du validation set
## 2. On récupère une probabilité
## 3. On calcule l'AUC
##
## Le meilleur k est celui qui donne la plus grande AUC.

k_values <- seq(1, 30, by = 2)

auc_results <- data.frame(
  k = k_values,
  AUC = NA
)


for (i in seq_along(k_values)) {
  
  k <- k_values[i]
  
  ## Prédiction avec le KNN
  
  pred <- knn(
    train = x_train,
    test = x_valid,
    cl = y_train,
    k = k,
    prob = TRUE
  )
  
  
  ## prob = TRUE renvoie la proportion de voisins
  ## appartenant à la classe prédite.
  
  probs <- attr(pred, "prob")
  
  
  ## On transforme cette probabilité en probabilité
  ## d'appartenir à la classe DIFF = 1.
  
  probs_class1 <- ifelse(
    pred == "1",
    probs,
    1 - probs
  )
  
  
  ## Calcul de la courbe ROC
  
  roc_obj <- roc(
    y_valid,
    probs_class1,
    quiet = TRUE
  )
  
  
  ## On enregistre l'AUC
  
  auc_results$AUC[i] <- as.numeric(
    auc(roc_obj)
  )
}


## Affichage des AUC

print(auc_results)


## ============================================================
## 6. Graphique de l'AUC en fonction de k
## ============================================================

plot(
  auc_results$k,
  auc_results$AUC,
  type = "b",
  pch = 19,
  col = "steelblue",
  xlab = "k (nombre de voisins)",
  ylab = "AUC",
  main = "AUC en fonction de k (validation set)"
)


## Recherche du meilleur k

best_k <- auc_results$k[
  which.max(auc_results$AUC)
]


cat(
  "Meilleur k trouvé :",
  best_k,
  "avec AUC =",
  max(auc_results$AUC),
  "\n"
)


## ============================================================
## 7. Modèle final avec le meilleur k
## ============================================================

pred_best <- knn(
  train = x_train,
  test = x_valid,
  cl = y_train,
  k = best_k,
  prob = TRUE
)


## Récupération des probabilités

probs_best <- attr(
  pred_best,
  "prob"
)


## Probabilité d'appartenir à DIFF = 1

probs_best_class1 <- ifelse(
  pred_best == "1",
  probs_best,
  1 - probs_best
)


## Calcul de la ROC

roc_best <- roc(
  y_valid,
  probs_best_class1,
  quiet = TRUE
)


## ============================================================
## 8. Courbe ROC
## ============================================================

plot(
  roc_best,
  col = "steelblue",
  lwd = 2,
  main = paste0(
    "Courbe ROC - KNN (k=",
    best_k,
    ")"
  )
)

abline(
  a = 1,
  b = -1,
  lty = 2,
  col = "gray"
)


## AUC

cat(
  "AUC (validation) :",
  round(auc(roc_best), 3),
  "\n"
)


## ============================================================
## 9. Matrice de confusion
## ============================================================

conf_matrix <- table(
  Prédit = pred_best,
  Réel = y_valid
)

print(conf_matrix)


## ============================================================
## 10. Métriques
## ============================================================

VP <- conf_matrix["1", "1"]
VN <- conf_matrix["0", "0"]
FP <- conf_matrix["1", "0"]
FN <- conf_matrix["0", "1"]


sensibilite <- VP / (VP + FN)

specificite <- VN / (VN + FP)

accuracy <- (VP + VN) / sum(conf_matrix)


cat(
  "Sensibilité :",
  round(sensibilite, 3),
  "\n"
)

cat(
  "Spécificité :",
  round(specificite, 3),
  "\n"
)

cat(
  "Accuracy    :",
  round(accuracy, 3),
  "\n"
)


## ============================================================
## 11. Robustesse : plusieurs partitions aléatoires
## ============================================================
## On répète le split 10 fois.
##
## Le k reste celui trouvé précédemment.
##
## On calcule une AUC pour chaque partition,
## puis on calcule la moyenne et l'écart-type.

n_repeats <- 10

auc_repeats <- numeric(n_repeats)


for (r in 1:n_repeats) {
  
  set.seed(r)
  
  idx_r <- sample(
    1:n,
    size = 0.8 * n
  )
  
  
  train_r <- data[idx_r, ]
  
  valid_r <- data[-idx_r, ]
  
  
  ## Moyennes et écarts-types du train
  
  means_r <- sapply(
    train_r[, vars_num],
    mean
  )
  
  sds_r <- sapply(
    train_r[, vars_num],
    sd
  )
  
  
  ## Standardisation du train
  
  x_train_r <- standardize(
    train_r[, vars_num],
    means_r,
    sds_r
  )
  
  
  ## Standardisation du validation
  
  x_valid_r <- standardize(
    valid_r[, vars_num],
    means_r,
    sds_r
  )
  
  
  ## Prédiction
  
  pred_r <- knn(
    train = x_train_r,
    test = x_valid_r,
    cl = train_r$DIFF,
    k = best_k,
    prob = TRUE
  )
  
  
  ## Probabilités
  
  probs_r <- attr(
    pred_r,
    "prob"
  )
  
  
  probs_r_class1 <- ifelse(
    pred_r == "1",
    probs_r,
    1 - probs_r
  )
  
  
  ## AUC
  
  roc_r <- roc(
    valid_r$DIFF,
    probs_r_class1,
    quiet = TRUE
  )
  
  
  auc_repeats[r] <- as.numeric(
    auc(roc_r)
  )
}


## Résultats

cat(
  "AUC moyenne sur",
  n_repeats,
  "partitions aléatoires (k =",
  best_k,
  ") :",
  round(mean(auc_repeats), 3),
  "\n"
)

cat(
  "Écart-type de l'AUC sur ces partitions :",
  round(sd(auc_repeats), 3),
  "\n"
)


## ============================================================
## 12. VISUALISATION DU KNN AVEC UNE PCA
## ============================================================
## ATTENTION :
##
## Cette PCA ne sert PAS à entraîner le KNN.
##
## Le KNN a bien été entraîné avec toutes les variables
## sélectionnées dans vars_num.
##
## La PCA sert uniquement à représenter toutes ces variables
## dans un graphique en 2 dimensions.


## ------------------------------------------------------------
## 12.1 PCA sur le train
## ------------------------------------------------------------
## On réalise la PCA uniquement sur le train.
##
## C'est important car le validation set ne doit pas
## influencer la transformation utilisée pour représenter
## les données.

pca <- prcomp(
  x_train,
  center = FALSE,
  scale. = FALSE
)


## ------------------------------------------------------------
## 12.2 Projection du train et du validation
## ------------------------------------------------------------

pca_train <- as.data.frame(
  predict(pca, x_train)
)

pca_valid <- as.data.frame(
  predict(pca, x_valid)
)


## ------------------------------------------------------------
## 12.3 Création d'un tableau pour le graphique
## ------------------------------------------------------------

pca_train$jeu <- "Train"

pca_valid$jeu <- "Validation"


pca_train$classe_reelle <- as.character(
  y_train
)

pca_valid$classe_reelle <- as.character(
  y_valid
)


pca_valid$prediction <- as.character(
  pred_best
)


## Pour les observations du train,
## on ne fait pas de prédiction ici.

pca_train$prediction <- NA


## ------------------------------------------------------------
## 12.4 Identifier les erreurs du KNN
## ------------------------------------------------------------
## Pour le validation set :
##
## TRUE  = mauvaise classification
## FALSE = bonne classification

pca_valid$erreur <- (
  pca_valid$classe_reelle !=
    pca_valid$prediction
)


## ------------------------------------------------------------
## 12.5 Graphique PCA
## ------------------------------------------------------------

plot(
  pca_train$PC1,
  pca_train$PC2,
  pch = 16,
  col = ifelse(
    pca_train$classe_reelle == "1",
    "steelblue",
    "gray"
  ),
  xlab = "Composante principale 1",
  ylab = "Composante principale 2",
  main = paste0(
    "KNN - Visualisation PCA du split 80/20 (k=",
    best_k,
    ")"
  )
)


## Ajout des observations du validation set

points(
  pca_valid$PC1,
  pca_valid$PC2,
  pch = 17,
  col = ifelse(
    pca_valid$erreur,
    "red",
    ifelse(
      pca_valid$classe_reelle == "1",
      "steelblue",
      "gray"
    )
  )
)


## Légende

legend(
  "topright",
  legend = c(
    "Train - classe 0",
    "Train - classe 1",
    "Validation - correctement classé",
    "Validation - erreur"
  ),
  pch = c(16, 16, 17, 17),
  col = c(
    "gray",
    "steelblue",
    "steelblue",
    "red"
  )
)


## ============================================================
## FIN
##
## Le modèle :
## - utilise toutes les variables sauf DIFF, AGE, R22 et R8
## - utilise un split aléatoire 80/20
## - standardise à partir du train uniquement
## - choisit k en maximisant l'AUC
## - calcule ROC, AUC, matrice de confusion,
##   sensibilité, spécificité et accuracy
## - teste la robustesse sur 10 partitions
##
## La PCA à la fin sert uniquement à visualiser
## les données et les erreurs du KNN.
## ============================================================
```
