# Ce fichier nous permet de gerer LES traces et de tester les fonctions de notre packages.
# Il ne fait pas parti du package. usethis::use_build_ignore("devtools_history.R").

usethis::use_build_ignore("devtools_history.R")
usethis::use_package("stats")
usethis::use_package("magrittr")
usethis::use_package("plyr")
usethis::use_package("parallel")
usethis::use_package("dplyr")
usethis::use_package("utils")
usethis::use_package("plotROC")
usethis::use_package("numDeriv")
usethis::use_package("klaR")
usethis::use_package("PCAmixdata")
usethis::use_package("tidytable")

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
    select(-obs_err) %>%
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
n <-100000  # Number of obervations
p <- 5 # Number of variables
theta = runif(p+1) # Theta vector
X <- cbind(1,matrix(rnorm(n*p),n,p))
Z <- X %*% theta # linear combination of variables
fprob <- ifelse(Z<0, exp(Z)/(1+exp(Z)),1/(1+exp(-Z))) # Calcul des probas d'affectation
y<- rbinom(n,1,fprob)
data = as.data.frame(cbind(X,y))
data$V1 <- NULL # delete colomn de biais. It will be created when fit is called


library(dgrGlm)
sigmoid(Z)
logLoss(theta,X,y)
gradient(theta,X,y)

# -----------------------------------------------------------------------------------------

#TEST FIT

# BATCH SEQUENTIEL
model_batch_seq <- dgrglm.fit(y~., data = data, mode_compute="sequentiel",
                              leaning_rate=0.1, max_iter=1000,tolerance=1e-06)
# SURCHARGE PRINT ET SUMMARY
summary(model_batch_seq)
print(model_batch_seq)

# MINI BATCH SEQUENTIEL
model_minibatch_seq <- dgrglm.fit(y~., data = data, mode_compute="sequentiel",leaning_rate=0.1,
                                  max_iter=1000,tolerance=1e-06,batch_size = 10)
# SURCHARGE PRINT ET SUMMARY
summary(model_batch_seq)
print(model_batch_seq)

# ONLINE SEQUENTIEL
model_online_seq <- dgrglm.fit(y~., data = data, mode_compute="sequentiel",
                               leaning_rate=0.1, max_iter=1000,tolerance=1e-06, batch_size = 1)
summary(model_batch_seq)
print(model_batch_seq)

# COMPARAISON MODELE
perf <- compare_model(probas_mod1=model_batch_seq$probas,
                      probas_mod2=model_online_seq$probas,
                      y=model_online_seq$y_val[,1])
perf$toPlo

# SEQUENTIEL EXECUTION TIME
print(system.time(model_batch_seq <- dgrglm.fit(y~., data = data, mode_compute="sequentiel",
                                                leaning_rate=0.1, max_iter=1000,tolerance=1e-04)))


print(system.time(model_minibatch_seq <- dgrglm.fit(y~., data = data, mode_compute="sequentiel",
                                                    batch_size = 100,leaning_rate=0.1,
                                                    max_iter=1000,tolerance=1e-04)))
# FONCTION SUMMARY ET PRINT SURCHARGÉ
print(model_batch_seq2)
summary(model_batch_seq2)

# FOR PREDICT
new_data<-data[,model_batch_seq$explicatives]
predict<- dgrglm.predict(model_batch_seq,new_data, type_pred = 'CLASS')
predict$new_data_classify

# MICROBENCHMARK SEQUENTIEL
library(microbenchmark)
microbenchmark(
  model_batch_seq <- dgrglm.fit(y~., data = data, mode_compute="sequentiel",
                                leaning_rate=0.1, max_iter=500,tolerance=1e-04),
  model_minibatch_seq <- dgrglm.fit(y~., data = data, mode_compute="sequentiel",leaning_rate=0.1,
                                    max_iter=500,tolerance=1e-04,batch_size = 10),
  model_online_seq <- dgrglm.fit(y~., data = data, mode_compute="sequentiel",
                                 leaning_rate=0.1, max_iter=500,tolerance=1e-04, batch_size = 1),
  times = 1,
  unit = "s"
)


# PARALLEL EXECUTION

# PARALLEL BATCH
model_batch_parallel <- dgrglm.fit(y~., data = data, ncores=3, mode_compute="parallel",
                                   leaning_rate=0.1, max_iter=100,tolerance=1e-06)
# SURCHARGE PRINT ET SUMMARY
print(model_batch_parallel)
summary(model_batch_parallel)

# PARALLEL MINI BATCH
model_minibatch_parallel <- dgrglm.fit(y~., data = data, ncores=3, mode_compute="parallel",
                                       leaning_rate=0.1, max_iter=100,tolerance=1e-06,batch_size = 10)
# SURCHARGE PRINT ET SUMMARY
print(model_minibatch_parallel)
summary(model_minibatch_parallel)

# PARALLEL ONLINE
model_online_parallel <- dgrglm.fit(y~., data = data, ncores=3, mode_compute="parallel",
                                    leaning_rate=0.1, max_iter=1000,tolerance=1e-06,batch_size = 1)

# SURCHARGE PRINT ET SUMMARY
print(model_online_parallel)
summary(model_online_parallel)

# MICROBENCHMARK PARALLEL
library(microbenchmark)
microbenchmark(
  model_batch_parallel <- dgrglm.fit(y~., data = data, ncores=3, mode_compute="parallel",
                                     leaning_rate=0.1, max_iter=500,tolerance=1e-04),

  model_minibatch_parallel <- dgrglm.fit(y~., data = data, ncores=3, mode_compute="parallel",
                                         leaning_rate=0.1, max_iter=500,tolerance=1e-04,batch_size = 10),
  model_online_parallel <- dgrglm.fit(y~., data = data, ncores=3, mode_compute="parallel",
                                      leaning_rate=0.1, max_iter=500,tolerance=1e-04,batch_size = 1),
  times = 1,
  unit = "s"
)

# COMPARAISON MODEL
perf1 <- compare_model(probas_mod1=model_batch_parallel$probas,
                      probas_mod2=model_minibatch_parallel$probas,
                      y=model_minibatch_parallel$y_val[,1])
perf1$toPlo

perf2 <- compare_model(probas_mod1=model_batch_parallel$probas,
                      probas_mod2=model_online_parallel$probas,
                      y=model_online_parallel$y_val[,1])
perf2$toPlo

perf3 <- compare_model(probas_mod1=model_minibatch_parallel$probas,
                      probas_mod2=model_online_parallel$probas,
                      y=model_online_parallel$y_val[,1])
perf3$toPlo


# EXECUTION TIME
print(system.time(model_batch_parallel <- dgrglm.fit(y~., data = data, mode_compute="parallel",
                                                     leaning_rate=0.1, max_iter=100,tolerance=1e-06, centering = TRUE)))



print(model_batch_parallel)
summary(model_batch_parallel)





print(system.time(model_minibatch_parallel <- dgrglm.fit(y~., data = data, ncores=3, mode_compute="parallel",
                                                         batch_size = 100,leaning_rate=0.1, max_iter=3000,
                                                         tolerance=1e-06)))

print(system.time(model_online_parallel <- dgrglm.fit(y~., data = data, ncores=3, mode_compute="parallel",
                                                      batch_size = 1,leaning_rate=0.1, max_iter=100,
                                                      tolerance=1e-06)))




