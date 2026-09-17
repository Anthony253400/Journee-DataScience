# ============================================================
# 1. CHARGEMENT DES DONNEES D'ENTRAINEMENT
# ============================================================

data <- read.csv(file.choose())

# Vérification des données
head(data)
dim(data)
str(data)
summary(data)
names(data)

# Valeurs manquantes et doublons
colSums(is.na(data))
sum(duplicated(data))

# Répartition de la variable cible
table(data$DIFF)

# DIFF est une variable catégorielle binaire : 0 ou 1
data$DIFF <- as.factor(data$DIFF)

str(data)

barplot(
  table(data$DIFF),
  main = "Répartition de DIFF",
  xlab = "Classe DIFF",
  ylab = "Nombre d'observations"
)


# ============================================================
# 2. ANALYSE EXPLORATOIRE
# ============================================================

# Analyse de TOF
table(data$TOF)
table(data$TOF, data$DIFF)
prop.table(table(data$TOF, data$DIFF), margin = 1)

# Comparaison des variables selon DIFF
par(mfrow = c(2, 3))

boxplot(AGE ~ DIFF, data = data,
        main = "AGE selon DIFF",
        xlab = "DIFF", ylab = "AGE")

boxplot(R7 ~ DIFF, data = data,
        main = "R7 selon DIFF",
        xlab = "DIFF", ylab = "R7")

boxplot(R8 ~ DIFF, data = data,
        main = "R8 selon DIFF",
        xlab = "DIFF", ylab = "R8")

boxplot(R17 ~ DIFF, data = data,
        main = "R17 selon DIFF",
        xlab = "DIFF", ylab = "R17")

boxplot(R22 ~ DIFF, data = data,
        main = "R22 selon DIFF",
        xlab = "DIFF", ylab = "R22")

boxplot(R32 ~ DIFF, data = data,
        main = "R32 selon DIFF",
        xlab = "DIFF", ylab = "R32")

par(mfrow = c(1, 1))

# Corrélations entre variables numériques
cor(data[, c("AGE", "R7", "R8", "R17", "R22", "R32")])

# Étude des valeurs extrêmes de R22
sort(data$R22, decreasing = TRUE)[1:10]

hist(
  data$R22,
  main = "Distribution de R22",
  xlab = "R22"
)


# ============================================================
# 3. SEPARATION TRAIN / VALIDATION
# ============================================================

# Fixer le hasard pour obtenir le même découpage
set.seed(123)

# Sélection aléatoire de 80 % des observations
indices <- sample(1:nrow(data), 0.8 * nrow(data))

# 80 % entraînement / 20 % validation
train <- data[indices, ]
valid <- data[-indices, ]

# Vérification
dim(train)
dim(valid)

table(train$DIFF)
table(valid$DIFF)


# ============================================================
# 4. REGRESSION LOGISTIQUE
# ============================================================

modele_log <- glm(
  DIFF ~ TOF + AGE + R7 + R8 + R17 + R22 + R32,
  data = train,
  family = binomial
)

# Résultats du modèle
summary(modele_log)


# ============================================================
# 5. PREDICTIONS SUR LA VALIDATION
# ============================================================

proba_log <- predict(
  modele_log,
  newdata = valid,
  type = "response"
)

head(proba_log)
length(proba_log)


# ============================================================
# 6. COURBE ROC ET AUC
# ============================================================

# À installer uniquement la première fois :
# install.packages("pROC")

library(pROC)

roc_log <- roc(valid$DIFF, proba_log)

# AUC
auc(roc_log)

# Courbe ROC
plot(
  roc_log,
  main = "Courbe ROC - Régression logistique"
)


# ============================================================
# 7. MATRICES DE CONFUSION SELON LE SEUIL
# ============================================================

# Seuil = 0.3
pred_03 <- ifelse(proba_log >= 0.3, 1, 0)

matrice_03 <- table(
  Reel = valid$DIFF,
  Predit = pred_03
)

matrice_03


# Seuil = 0.5
pred_05 <- ifelse(proba_log >= 0.5, 1, 0)

matrice_05 <- table(
  Reel = valid$DIFF,
  Predit = pred_05
)

matrice_05


# Seuil = 0.7
pred_07 <- ifelse(proba_log >= 0.7, 1, 0)

matrice_07 <- table(
  Reel = valid$DIFF,
  Predit = pred_07
)

matrice_07


# ============================================================
# 8. CHARGEMENT DU FICHIER TEST
# ============================================================

test <- read.csv(file.choose())

dim(test)
str(test)


# ============================================================
# 9. MODELE FINAL SUR LES 400 OBSERVATIONS
# ============================================================

modele_final <- glm(
  DIFF ~ TOF + AGE + R7 + R8 + R17 + R22 + R32,
  data = data,
  family = binomial
)


# ============================================================
# 10. PREDICTIONS SUR LES 120 OBSERVATIONS TEST
# ============================================================

proba_test <- predict(
  modele_final,
  newdata = test,
  type = "response"
)

head(proba_test)

# Transformer les probabilités en classes avec seuil 0.5
prediction_test <- ifelse(proba_test >= 0.5, 1, 0)


# ============================================================
# 11. CREATION DU FICHIER DE SOUMISSION
# ============================================================

soumission <- data.frame(
  ID = 1:120,
  DIFF = prediction_test
)

head(soumission)

write.csv(
  soumission,
  "reda-soumission_reg_log.csv",
  row.names = FALSE
)

# Vérifier que le fichier existe
file.exists("soumission_reg_log.csv")

# Afficher le dossier où il est enregistré
getwd()
