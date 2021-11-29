usethis::use_build_ignore("devtools_history.R")
usethis::use_package("stats")
usethis::use_package("magrittr") # à executer plus tard
usethis::use_package("plyr")
usethis::use_package("parallel")
usethis::use_package("dplyr")
usethis::use_package("utils")
usethis::use_package("plotROC")
usethis::use_package("numDeriv")
usethis::use_package("klaR")

library(tidyverse)
library(plotROC)

# FONCTION UTILITAIRE POUR COMPARER DEUX MODÉLE AVEC UNE COURBE ROC
compare_model<- function(probas_mod1, probas_mod2, y){
  predProbas <- data.frame(model1=probas_mod1, model2=probas_mod2)
  # Estimation des classes en fonctions des probas
  predClass <- apply(predProbas >= 0.5, 2, factor, labels=c(0,1))
  predClass <- data.frame(predClass)
  # Erreur de classification des deux modéles
  df_err<- predClass %>%
    mutate(obs=y) %>%
    summarise_all(funs(err=mean(obs!=.))) %>%
    #select(-obs_err) %>%
    round(3)
  # Etude des courbes ROC
  df_roc <- predProbas %>%
    mutate(obs=y) %>%
    gather(key = methode, value=score, model1, model2)

  toPlot<- ggplot(df_roc)+
    aes(d=obs,m=score, color=methode)+
    geom_roc()+
    theme_classic()
  return(list(predProbas = predProbas, PredClass = predClass, models_error = df_err,  toPlot=toPlot))
}


# CODE DE GENERATION DONNÉES LOGISTIQUE
set.seed(103)
n <-1000
p <- 10
theta = runif(p+1) # or theta = runif(7)
X <- cbind(1,matrix(rnorm(n*p),n,p)) #  6 Variables quantitative
X1 = matrix(rnorm(n*p),n,p)
X1 <- as.data.frame(X1)
Z <- X %*% theta # combinaison lineare de variable
fprob <- ifelse(Z<0, exp(Z)/(1+exp(Z)),1/(1+exp(-Z))) # Calcul des probas d'affectation
y<- rbinom(n,1,fprob)
data = as.data.frame(cbind(y,X1))


# TESTE DES FONCTIONS

# PARALLÉLISATION DES CALCULS
library(xlsx)
data <- read.xlsx(file="~/Desktop/Lyon2/SISE/AtelierMachLeraning/Reg Logistique_opt_hyp/ionosphere.xlsx",sheetIndex=1,header=T)
data = data[,-33]

library(dgrGlm)
sigmoid(Z)
logLoss(theta,X,y)
gradient(theta,X,y)

# -----------------------------------------------------------------------------------------
#TEST FIT

# SEQUENTIEL
print(system.time(model_batch_seq <- dgrglm.fit(y~., data = data, mode_compute="sequentiel",
                                                leaning_rate=0.1, max_iter=2000,tolerance=1e-06,
                                                feature_selection=TRUE, p_value=0.01)))
# FONCTION SUMMARY ET PRINT SURCHARGÉ
print(model_batch_seq)
summary(model_batch_seq)

print(system.time(model_batch_seq2 <- dgrglm.fit(y~., data = data, mode_compute="sequentiel",
                                                leaning_rate=0.1, max_iter=2000,tolerance=1e-06,
                                                iselasticnet=TRUE, C=10, rho=1)))
# FONCTION SUMMARY ET PRINT SURCHARGÉ
print(model_batch_seq2)
summary(model_batch_seq2)

new_data<-data[,model_batch_seq$explicatives]
predict$new_data_classify<- dgrglm.predict(model_batch_seq,new_data)

print(predict)
summary(predict)
predict$new_data_classify

print(system.time(model_minibatch_seq <- dgrglm.fit(y~., data = data, mode_compute="sequentiel",batch_size = 100, leaning_rate=0.1, max_iter=2000,tolerance=1e-06)))
print(system.time(model_online_seq <- dgrglm.fit(y~., data = data, mode_compute="sequentiel",batch_size = 1, leaning_rate=0.1, max_iter=100,tolerance=1e-06)))



# PARALLEL
print(system.time(model_batch_parallel <- dgrglm.fit(y~., data = data, ncores=3, mode_compute="parallel",
                                                     leaning_rate=0.1, max_iter=100,tolerance=1e-06,
                                                     iselasticnet=TRUE, C=10, rho=0.001)))
print(model_batch_parallel)
summary(model_batch_parallel)

print(system.time(model_minibatch_parallel <- dgrglm.fit(y~., data = data, ncores=3, mode_compute="parallel",
                                                         batch_size = 100,leaning_rate=0.1, max_iter=3000,tolerance=1e-06)))

print(system.time(model_online_parallel <- dgrglm.fit(y~., data = data, ncores=3, mode_compute="parallel",
                                                      batch_size = 1,leaning_rate=0.1, max_iter=100,tolerance=1e-06)))


# EVALUATE PERFORMANCE MODEL
perf <- compare_model(probas_mod1=model_minibatch_parallel$probas,
                      probas_mod2=model_online_parallel$probas,
                      y=model_online_parallel$y_val[,1])
perf$toPlot



