usethis::use_build_ignore("devtools_history.R")
usethis::use_package("stats")
usethis::use_package("magrittr") # à executer plus tard
usethis::use_package("plyr")
usethis::use_package("parallel")
usethis::use_package("dplyr")
usethis::use_package("utils")
usethis::use_package("plotROC")
usethis::use_package("ggplot2")

library(tidyverse)
library(plotROC)

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


# GENERATION DONNÉES LOGISTIQUE
set.seed(103)
n <-100
p <- 10

theta = runif(p+1) # or theta = runif(7)
X <- cbind(1,matrix(rnorm(n*p),n,p)) #  6 Variables quantitative
X1 = matrix(rnorm(n*p),n,p)
X1 <- as.data.frame(X1)
X1$biais <- 1
class(X1)
class(theta)
Z <- X %*% theta # combinaison lineare de variable
fprob <- ifelse(Z<0, exp(Z)/(1+exp(Z)),1/(1+exp(-Z))) # Calcul des probas d'affectation
y<- rbinom(n,1,fprob)
data = as.data.frame(cbind(y,X1))

leaning_rate = 0.1
max_iter = 1000
tolerance = 1e-04
batch_size = 10
batch_size_online = 1
random_state = 102
ncores = 3


library(dgrGlm)
sigmoid(Z)
logLoss(theta,X,y)
gradient(theta,X,y)

# Gradient sequentiel
print(system.time(model_batch_seq <- dg_batch_seq(X1,y,theta,leaning_rate=0.1, max_iter=100,tolerance=1e-06)))
xi=1:model_batch_seq$nbIter
yi=model_batch_seq$history_cost
plot(xi, yi, type="l")

print(system.time(model_mini_batch_seq <- dg_batch_minibatch_online_seq(X, y, theta, batch_size=10, random_state=102, leaning_rate=0.05, max_iter=100,tolerance=1e-06)))
xi=1:model_mini_batch_seq$nb_iter_for
yi=model_mini_batch_seq$history_cost
plot(xi, yi, type="l")

print(system.time(model_mini_online_batch_seq <- dg_batch_minibatch_online_seq(X, y, theta, batch_size=1, random_state=102, leaning_rate=0.05, max_iter=100,tolerance=1e-06)))
xi=1:model_mini_online_batch_seq$nb_iter_for
yi=model_mini_online_batch_seq$history_cost
plot(xi, yi, type="l")


Z = X %*% seq.coef_online_batch
prob_pred <- sigmoid(Z)
y_pred <- ifelse(prob_pred>0.5,1,0)
metric_R2(y=y,ypred = y_pred)

# Gradient parallél
print(system.time(model_batch_parallel <- dgsrow_batch_parallele(X1, y, theta, ncores=3, leaning_rate=0.05, max_iter=100,tolerance=1e-06)))
print(system.time(model_mini_online_batch_parallel <- dgs_minibatch_online_parallle(X1,y,theta,ncores=3,batch_size = 10,leaning_rate=0.05, max_iter=100,tolerance=1e-06)))
#print(system.time(model_mini_online_batch_parallel <- dgs_minibatch_online_parallle2(X1,y,theta,ncores,batch_size_online,random_state,leaning_rate, max_iter,tolerance)))

#TEST FIT

# SEQUENTIEL
print(system.time(model_batch_seq <- dgrglm.fit(y~., data = data, ncores=3, mode_compute="sequentiel",leaning_rate=0.1, max_iter=2000,tolerance=1e-06)))
print(system.time(model_minibatch_seq <- dgrglm.fit(y~., data = data, ncores=3, mode_compute="sequentiel",batch_size = 351, leaning_rate=0.1, max_iter=2000,tolerance=1e-06)))
print(system.time(model_online_seq <- dgrglm.fit(y~., data = data, ncores=3, mode_compute="sequentiel",batch_size = 1, leaning_rate=0.1, max_iter=100,tolerance=1e-06)))

# PARALLEL
print(system.time(model_batch_parallel <- dgrglm.fit(y~., data = data, ncores=3, mode_compute="parallel",leaning_rate=0.1, max_iter=1000,tolerance=1e-06)))
print(system.time(model_minibatch_parallel <- dgrglm.fit(y~., data = data, ncores=3, mode_compute="parallel",batch_size = 5,leaning_rate=0.1, max_iter=1000,tolerance=1e-06)))
print(system.time(model_online_parallel <- dgrglm.fit(y~., data = data, ncores=3, mode_compute="parallel",batch_size = 1,leaning_rate=0.1, max_iter=100,tolerance=1e-06)))


# EVALUATE PERFORMANCE MODEL
perf <- evaluate_performance(model_batch_parallel$probas,model_online_parallel$probas,model_batch_seq$y_val[,1])
perf$toPlot

model_batch_seq$y_val[,1]
model_batch_parallel$y_val[,1]


seq.coef_batch = model_batch_seq$theta_final
seq.coef_mini_batch = model_mini_batch_seq$theta
seq.coef_online_batch = model_mini_online_batch_seq$theta
parbatch.coef_batch_batch = model_batch_parallel$theta_final
parbatch.coef_mini_batch = model_mini_online_batch_parallel$theta_final
# Newton Raphson BFGS
newton.coef <- optim(theta, logLoss, y=y, X=as.matrix(X1), method = "BFGS")$par
# Comparaison des coefs
cbind(seq.coef_batch=seq.coef_batch, seq.coef_mini_batch =seq.coef_mini_batch , seq.coef_online_batch=seq.coef_online_batch,
      parbatch.coef_batch_batch = parbatch.coef_batch_batch, parbatch.coef_mini_batch=parbatch.coef_mini_batch, BFGS=newton.coef)



library(microbenchmark)
microbenchmark(
  dg_batch_seq(X,y,theta,leaning_rate, max_iter,tolerance),
  dg_batch_minibatch_online_seq(X,y,theta,batch_size,random_state,leaning_rate, max_iter,tolerance),
  times = 1,
  unit = "s"
)

xi=1:model_batch_seq$nbIter
yi=model_batch_seq$history_cost
plot(xi, yi, type="l")


ypred =binary_predict(sigmoid(Z))
metric_R2(y,ypred)


global_grad_descent(X,y,theta, batch_size=1, random_state=1, leaning_rate=0.1, max_iter=6000, tolerance=1e-04)

res = fit(y~x1+x2+x3+x4+x5+x6, as.data.frame(df), mode="MINI_BATCH",batch_size=10, random_state=1, leaning_rate=0.1, max_iter=6000, tolerance=1e-04)








# PARALLÉLISATION DES CALCULS
library(xlsx)
data <- read.xlsx(file="~/Desktop/Lyon2/SISE/AtelierMachLeraning/Reg Logistique_opt_hyp/ionosphere.xlsx",sheetIndex=1,header=T)
data = data[,-33]


# GENERATION DONNÉES LOGISTIQUE
set.seed(100)
n <- 500
p <- 20
theta = runif(p+1)
X <- cbind(1,matrix(rnorm(n*p),n,p))
X1 = matrix(rnorm(n*p),n,p)
Z <- X %*% theta
fprob <- ifelse(Z<0, exp(Z)/(1+exp(Z)),1/(1+exp(-Z)))
y<- rbinom(n,1,fprob)
data = as.data.frame(cbind(y,X1))

#df = model.frame(y~V2+V3, data = data)

print(system.time(modele <- dgrglm.fit(y~., data, ncores=3, random_state=1, leaning_rate=0.1, max_iter=100, batch_size = 1, tolerance=1e-04,centering = FALSE)))
modele$X # je verifie s'il a bien centrer et reduit
dim(modele)
modele$res$theta
modele

dgrglm.predict(modele, data[,-1])

xi=1:modele$res$nb_iter_while
yi=modele$res$history_cost
plot(xi, yi, type="l")



