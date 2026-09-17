## ============================================================
## KNN - Choix du k, courbe ROC et AUC
## ============================================================

# install.packages(c("class", "pROC"))
library(class)   # fonction knn()
library(pROC)    # calcul ROC / AUC

## --- 1. Chargement des données ---
data <- read.csv("C:/Users/bruno/Journee-DataScience/data/farms_train.csv", header = TRUE, sep = ",")
data$DIFF <- as.factor(data$DIFF)

## --- 2. Split train / validation (interne, pour choisir k et évaluer) ---
set.seed(42)  # pour la reproductibilité
n <- nrow(data)
idx <- sample(1:n, size = 0.8 * n)  # 80% train, 20% validation
train_set <- data[idx, ]
valid_set <- data[-idx, ]

vars_num <- setdiff(colnames(data), "DIFF")

## --- 3. Standardisation des variables (INDISPENSABLE pour KNN) ---
## KNN se base sur des distances : si les variables n'ont pas la même échelle,
## celles avec de grandes valeurs numériques dominent le calcul de distance.
## On calcule moyenne/écart-type SUR LE TRAIN UNIQUEMENT, puis on applique
## les mêmes paramètres au validation set (pour éviter la fuite de données).

train_means <- sapply(train_set[, vars_num], mean)
train_sds   <- sapply(train_set[, vars_num], sd)

standardize <- function(df, means, sds) {
  as.data.frame(scale(df, center = means, scale = sds))
}

x_train <- standardize(train_set[, vars_num], train_means, train_sds)
x_valid <- standardize(valid_set[, vars_num], train_means, train_sds)

y_train <- train_set$DIFF
y_valid <- valid_set$DIFF

## ============================================================
## 4. Choix du meilleur k via l'AUC
## ============================================================
## On teste une grille de valeurs de k, et pour chacune on calcule l'AUC
## sur le validation set. On garde le k qui maximise l'AUC.

k_values <- seq(1, 30, by = 2)  # k impairs pour éviter les ex-aequo
auc_results <- data.frame(k = k_values, AUC = NA)

for (i in seq_along(k_values)) {
  k <- k_values[i]
  
  # prob = TRUE renvoie la proportion de voisins de la classe majoritaire
  pred <- knn(train = x_train, test = x_valid, cl = y_train, k = k, prob = TRUE)
  
  # on récupère la probabilité associée à la classe prédite,
  # puis on la transforme en "probabilité d'être DIFF=1"
  probs <- attr(pred, "prob")
  probs_class1 <- ifelse(pred == "1", probs, 1 - probs)
  
  roc_obj <- roc(y_valid, probs_class1, quiet = TRUE)
  auc_results$AUC[i] <- as.numeric(auc(roc_obj))
}

print(auc_results)

## Visualisation de l'AUC en fonction de k
plot(auc_results$k, auc_results$AUC, type = "b", pch = 19, col = "steelblue",
     xlab = "k (nombre de voisins)", ylab = "AUC",
     main = "AUC en fonction de k (validation set)")

best_k <- auc_results$k[which.max(auc_results$AUC)]
cat("Meilleur k trouvé :", best_k, "avec AUC =", max(auc_results$AUC), "\n")

## ============================================================
## 5. Modèle final avec le meilleur k, courbe ROC et AUC détaillées
## ============================================================
pred_best <- knn(train = x_train, test = x_valid, cl = y_train, k = best_k, prob = TRUE)
probs_best <- attr(pred_best, "prob")
probs_best_class1 <- ifelse(pred_best == "1", probs_best, 1 - probs_best)

roc_best <- roc(y_valid, probs_best_class1, quiet = TRUE)

## Courbe ROC
plot(roc_best, col = "steelblue", lwd = 2,
     main = paste0("Courbe ROC - KNN (k=", best_k, ")"))
abline(a = 1, b = -1, lty = 2, col = "gray")  # diagonale de référence

## AUC
cat("AUC (validation) :", round(auc(roc_best), 3), "\n")

## Matrice de confusion (seuil 0.5 par défaut, via la classe prédite par knn)
conf_matrix <- table(Prédit = pred_best, Réel = y_valid)
print(conf_matrix)

## Métriques dérivées
VP <- conf_matrix["1", "1"]
VN <- conf_matrix["0", "0"]
FP <- conf_matrix["1", "0"]
FN <- conf_matrix["0", "1"]

sensibilite <- VP / (VP + FN)
specificite <- VN / (VN + FP)
accuracy <- (VP + VN) / sum(conf_matrix)

cat("Sensibilité :", round(sensibilite, 3), "\n")
cat("Spécificité :", round(specificite, 3), "\n")
cat("Accuracy    :", round(accuracy, 3), "\n")

## ============================================================
## 6. Robustesse : moyenne sur plusieurs partitions aléatoires
## ============================================================
## Un seul split train/validation peut donner un résultat qui dépend
## du hasard. On répète l'opération plusieurs fois et on moyenne l'AUC
## pour le k retenu, afin d'avoir une estimation plus fiable.

n_repeats <- 10
auc_repeats <- numeric(n_repeats)

for (r in 1:n_repeats) {
  set.seed(r)
  idx_r <- sample(1:n, size = 0.8 * n)
  train_r <- data[idx_r, ]
  valid_r <- data[-idx_r, ]
  
  means_r <- sapply(train_r[, vars_num], mean)
  sds_r   <- sapply(train_r[, vars_num], sd)
  
  x_train_r <- standardize(train_r[, vars_num], means_r, sds_r)
  x_valid_r <- standardize(valid_r[, vars_num], means_r, sds_r)
  
  pred_r <- knn(train = x_train_r, test = x_valid_r, cl = train_r$DIFF, k = best_k, prob = TRUE)
  probs_r <- attr(pred_r, "prob")
  probs_r_class1 <- ifelse(pred_r == "1", probs_r, 1 - probs_r)
  
  roc_r <- roc(valid_r$DIFF, probs_r_class1, quiet = TRUE)
  auc_repeats[r] <- as.numeric(auc(roc_r))
}

cat("AUC moyenne sur", n_repeats, "partitions aléatoires (k =", best_k, ") :",
    round(mean(auc_repeats), 3), "\n")
cat("Écart-type de l'AUC sur ces partitions :", round(sd(auc_repeats), 3), "\n")

## ============================================================
## FIN
## - Le k a été choisi automatiquement en maximisant l'AUC
## - On a vérifié la robustesse du résultat sur plusieurs partitions
## - La courbe ROC + AUC finale servent d'argument pour comparer KNN
##   aux autres méthodes (régression logistique, random forest...)
## ============================================================




## ============================================================
## 7. Prédiction sur le jeu de test pour Kaggle
## ============================================================

# Chargement du jeu de test
test <- read.csv("C:/Users/bruno/Journee-DataScience/data/farms_test.csv",
                 header = TRUE, sep = ",")

# Les variables utilisées pour le KNN
# On enlève DIFF car elle n'est pas connue dans le jeu de test
vars_num <- setdiff(colnames(data), "DIFF")

# ------------------------------------------------------------
# Standardisation
# ------------------------------------------------------------
# Cette fois, on entraîne le modèle sur TOUT le jeu train.
# On calcule donc les moyennes et écarts-types sur toutes les
# données d'apprentissage.

train_means_final <- sapply(data[, vars_num], mean)
train_sds_final   <- sapply(data[, vars_num], sd)

# Standardisation du train complet
x_train_final <- standardize(data[, vars_num],
                             train_means_final,
                             train_sds_final)

# Standardisation du jeu test avec les paramètres du train
x_test <- standardize(test[, vars_num],
                      train_means_final,
                      train_sds_final)

# Variable à prédire
y_train_final <- data$DIFF

# ------------------------------------------------------------
# KNN final
# ------------------------------------------------------------

pred_test <- knn(train = x_train_final,
                 test = x_test,
                 cl = y_train_final,
                 k = best_k)

# ------------------------------------------------------------
# Création du fichier de soumission
# ------------------------------------------------------------

submission <- data.frame(
  ID = 1:nrow(test),
  DIFF = as.numeric(as.character(pred_test))
)

# Vérification
head(submission)

# Sauvegarde
write.csv(submission,
          "C:/Users/bruno/Journee-DataScience/data/soumission_knn.csv",
          row.names = FALSE)

cat("Fichier de soumission créé !\n")
