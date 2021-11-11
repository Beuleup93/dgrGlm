####### Probléme de regression
#dataset <- CommViolPredUnnormalizedData <- read.csv("~/Downloads/CommViolPredUnnormalizedData.txt", header=FALSE, stringsAsFactors = TRUE)
dataset <- read.csv("~/Desktop/Lyon2/SISE/Projet R/VisaPremier.csv", stringsAsFactors=TRUE)
head(dataset)
str(dataset)
View(dataset)

library("VIM") # cette librairie permet de visualiser le dispositif de données manquante
# La fonction aggr représente le pourcentage de valeur manquante dans chaque variable
# Et nous avons aussi les combinaisons de variable qui ont des valeurs manquantes simultanés
# La combinaisons la plus fréquente est celle ou toute les variables sont observées, toutes les cases sont bleux(ligne du bas)
# Vu que le mécanisme de données manquante est aléatoire, nous pouvons:
# Soit l'imputation simple ou à valeur unique si l'objectif est de prévoir au mieu les valeurs manquantes.
dataset[dataset == '.'] <- NA
# Nombre de ligne avec des données manquantes
nrow(na.omit(dataset))
res<- summary(aggr(dataset, sortVar=TRUE))$combinations

# Remplacer les données manquantes par leurs moyenne
dataset$cartevp = factor(dataset$cartevp,
                           levels = c('Cnon', 'Coui'),
                           labels = c(0, 1))
dataset$sexe = factor(dataset$sexe,
                         levels = c('Sfem', 'Shom'),
                         labels = c(0, 1))

library(zoo)
library(DescTools)

dataset$nbpaiecb <- na.aggregate(as.integer(dataset$nbpaiecb), FUN = mean)
dataset$codeqlt <- na.aggregate(dataset$codeqlt, FUN = Mode)
dataset$departem <- na.aggregate(dataset$departem, FUN = Mode)
dataset$agemvt <- na.aggregate(dataset$agemvt, FUN = Mode)

reg.log = glm()

table(dataset$cartevp)
table(dataset$sexe)
table(dataset$sexer)
table(dataset$nbpaiecb)


library(tidyverse)
quantitative <- dataset

sep_var <- function(choix){

}
sep_var<- sapply(dataset, function(x) is.numeric(x) | is.integer(x))
quanti = colnames(dataset[sep_var==TRUE])
quali = colnames(dataset[sep_var==FALSE])

df1 = select(dataset, all_of(quali))
df2 = select(dataset, all_of(quanti))

df2 = df2[, c(-(ncol(df2)-1), -(ncol(df2)-2))]
data = select(dataset, colnames(df2), all_of(quali))
# SHUFLE
rows <- sample(nrow(data))
data <- data[rows, ]
# Caret
library(caret)
names(getModelInfo())

col = colnames(data)
data = data[, c(-(ncol(data)-2), -(ncol(data)-3), -(ncol(data)-4))]

# Create partition
set.seed(12)
trainIndex <- createDataPartition(data$cartevp, p=0.7, list= F)
print(length(trainIndex))

train = data[trainIndex,]
test = data[-trainIndex,]

# Cross Validation
train_control<- trainControl(method="cv", number=5)
reg.glm = train(cartevp~.,
                data=train,
                method="glm",
                maxit=1000,
                trControl=train_control)
print(reg.glm)
print(reg.glm$finalModel)

pred = predict(reg.glm, newdata = test)
print(table(pred))

# Matrice de confusion
mat<- ?confusionMatrix(reference=data$cartevp, data=pred, positive=1)
print(mat)







# Imputation des données manquantes par la moyenne
library(zoo)
for(i in 1:ncol(dataset)){
  if(is.numeric(dataset[,i])){
    #dataset[,i] = ifelse(is.na(dataset[,i]),
                         #ave(dataset[,i], FUN = function(x) mean(x, na.rm = TRUE)),
                        # dataset[,i])
    print(is.na(dataset[,i]))
  }
}


