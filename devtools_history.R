usethis::use_build_ignore("devtools_history.R")
usethis::use_package("stats")
usethis::use_package("magrittr") # à executer plus tard
usethis::use_package("plyr")
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

#create model
df = cbind(y,X1)
colnames(df) <- c("y","x1","x2","x3","x4","x5","x6")
head(df)
class(df)

res = fit(y~x1+x2+x3+x4+x5+x6, as.data.frame(df), mode="MINI_BATCH",batch_size=10, random_state=1, leaning_rate=0.1, max_iter=6000, tolerance=1e-04)
# Newton
newton.coef <- optim(theta, logLoss, y=y, X=X, method = "BFGS")$par
# Comparaison
cbind(GradDescBatch=res$res$theta, BFGS=newton.coef)
res$res$theta
newton.coef

# Formule
a = c(1,2,3)
b = c(3,5,7)
df = data.frame(a,b)
library(modelr)
dio = model_matrix(data, y ~ .)
theta = runif(ncol(dio))
length(theta)





# PARALLÉLISATION DES CALCULS
library(xlsx)
data <- read.xlsx(file="~/Desktop/Lyon2/SISE/AtelierMachLeraning/Reg Logistique_opt_hyp/ionosphere.xlsx",sheetIndex=1,header=T)
data = data[,-33]


# GENERATION DONNÉES LOGISTIQUE
set.seed(100)
n <- 10000
p <- 20
theta = runif(p+1)
X <- cbind(1,matrix(rnorm(n*p),n,p))
X1 = matrix(rnorm(n*p),n,p)
Z <- X %*% theta
fprob <- ifelse(Z<0, exp(Z)/(1+exp(Z)),1/(1+exp(-Z)))
y<- rbinom(n,1,fprob)
data = as.data.frame(cbind(y,X1))

#df = model.frame(y~V2+V3, data = data)

#data$fold = NULL
print(system.time(modele <- dgrglm.fit(y~., data, mode="MINI_BATCH", random_state=1, leaning_rate=0.1, max_iter=3000, batch_size = 10, tolerance=1e-04,centering = FALSE)))
modele$X # je verifie s'il a bien centrer et reduit
dim(modele)
modele$res$theta
modele

dgrglm.predict(modele, data[,-1])

xi=1:modele$res$nb_iter_while
yi=modele$res$history_cost
plot(xi, yi, type="l")



#_____________________________ CROSS VALIDATION ________________________________________________________________________________

# Parallélisation des calculs
set.seed(20)
# creation d'une variable contenant le numero des blocs(choisit)
data$fold = sample(1:4, nrow(data), replace = TRUE)
# repartition de la variable cible dans chaque groupe
table(data$y, data$fold)

# Cas ou on cherche le taux d'erreur moyenne des validations croisées
cross_validation_rgl <- function(data_app, data_val){
  modele <- dgrglm.fit(y~., data = data_app, mode="MINI_BATCH",centering = TRUE,leaning_rate=0.1, max_iter=3000, batch_size = 10, tolerance=1e-04)
  trueY = data_val$y
  data_val$y = NULL
  pred = dgrglm.predict(modele,new_data = data_val)
  y_val <- data.frame(reelY=trueY, yPred=pred$binary_predict)
  list(rl = modele, y_val, err_rate=mean(y_val$reelY != y_val$yPred))
}


# AVEC PARALELE
require(parallel)
cl<-makeCluster(detectCores())
clusterSetRNGStream(cl, iseed=78)

clusterExport(cl,varlist = c("data","cross_validation_rgl"))
clusterEvalQ(cl, {require(dgrGlm)})

print(system.time(res <- clusterApply(cl=cl, x=1:4, fun=function(fold){
  data_app = data[data$fold != fold, ] # creation apprentissage
  data_val = data[data$fold == fold, ] # creation validation avec suppression du y
  data_app$fold = NULL # suppression fold
  data_val$fold = NULL # suppression fold
  cross_validation_rgl(data_app,data_val)
})))

stopCluster(cl)
sapply(res, function(x) x$err_rate)
mean(sapply(res, function(x) x$err_rate))
sapply(res, function(x) x$rl$res$theta)
apply(sapply(res, function(x) x$rl$res$theta), 1,mean) # Estimation


# AVEC FOREACH ET DOPARALLELE
require(foreach)
require(doParallel)
cl <- makeCluster(detectCores()-1)
clusterSetRNGStream(cl,iseed=78)
registerDoParallel(cl)
print(system.time(res <-foreach(fold=1:4, .packages="dgrGlm",
              .noexport = setdiff(ls(),c("data","cross_validation_rgl"))) %dopar% {
                data_app = data[data$fold != fold, ] # creation apprentissage
                data_val = data[data$fold == fold, ] # creation validation avec suppression du y
                data_app$fold = NULL # suppression fold
                data_val$fold = NULL # suppression fold
                cross_validation_rgl(data_app,data_val)
              }))
stopCluster(cl)
sapply(res, function(x) x$err_rate)





