usethis::use_build_ignore("devtools_history.R")
usethis::use_package("stats")
usethis::use_package("magrittr") # à executer plus tard
usethis::use_package("plyr")
usethis::use_package("parallel")
usethis::use_package("dplyr")
usethis::use_package("rlang", type = "Imports")


# GENERATION DONNÉES LOGISTIQUE
set.seed(100)
n <- 40000
p <- 20

theta = runif(p+1) # or theta = runif(7)
X <- cbind(1,matrix(rnorm(n*p),n,p)) #  6 Variables quantitative
X1 = matrix(rnorm(n*p),n,p)
class(X1)
class(theta)
Z <- X %*% theta # combinaison lineare de variable
fprob <- ifelse(Z<0, exp(Z)/(1+exp(Z)),1/(1+exp(-Z))) # Calcul des probas d'affectation
y<- rbinom(n,1,fprob)
data = as.data.frame(cbind(y,X1))
class(data)

library(dgrGlm)
sigmoid(Z)
logLoss(theta,X,y)
gradient(theta,X,y)
grad_desc = grad_descent_batch(X,y,theta)
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



