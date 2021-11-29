# dgrGlm: GRadient Descent For Logistic Regression

### DESCRIPTION

This project is part of our training in Data Science at the University of Lyon 2.  The main objective is to develop an R package under S3 that allows to do binary 
logistic regression distributed on the different cores of the user's computer. Gradient descent is used for the optimization of the parameters. 
Here the idea is to allow the user to take advantage of the totality of these computer resources, if of course this can help to make the execution 
of the calculations faster. Here are the different functionalities of our package that we will present in the following lines:

* Binary logistic regression model in **sequential** mode 
  - Splitting of calculations and data for **parallel** execution
  - Binary logistic regression model in parallel mode
  - Comparison of model predictions 
  - Automatic Features selection
  - Model Microbenchmark
  - **Elasticnet** (Ridge and Lasso)
* Multinomiale logistic regression 
  - Recoding of the target variable (One vs All)

### Logistic Regression

Logistic regression is an old and well-known statistical predictive method that applies to binary classification, but can be easily extended to the multiclass 
framework (multinomial regression in particular, but not only...).It has become popular in recent years in machine learning thanks to "new" computational 
algorithms (optimization) and powerful libraries.
The work on regularization makes its use efficient in the context of high dimensional learning (nb. Var. Expl. >> nb. Obsv.). It is related to neural networks 
(simple perceptron).<br/>
The objective of this guide is not to explain the mathematical formulas around logistic regression but to give you an overview of our package.

### Installation and data loading

In order to use our package, you should install it from Github.

```sh
library(devtools)
install_github("Beuleup93/dgrGlm")
```

Once the package is downloaded and successfully installed, please load it for use.

```sh
library(dgrGlm)
```
<br/>

Now you can access all available functions of the package. To prove it, we will display the documentation of our fit function.
you can write in your console: **?dgrGlm.fit** to see the documentation or: 

```sh
help(dgrGlm.fit)
```
<br/>
<div align="center">
<img width="1121" alt="Capture d’écran 2021-11-29 à 15 57 01" src="https://user-images.githubusercontent.com/31353252/143900558-aa1d08e5-04a9-4942-8e0a-f528a779b5b4.png">
</div>
<br/>

In order to test our functions, we will work with the dataset **ionosphere.xlsx**. It consists of 351 obervations and 34 variables.

<div align="center">
<img width="490" alt="Capture d’écran 2021-11-29 à 15 59 47" src="https://user-images.githubusercontent.com/31353252/143900971-229feff3-18f8-4cba-89ce-0efcba432037.png">
</div>
<br/>

### 1. Binary Logistic Regression

We will start by testing the binary logistic regression on our dataset. The variable to be explained is Y and the explicatives variables are a03,...,a34.

#### General function fit
```sh
dgrglm.fit <- function(formule, data, ncores=NA, mode_compute="parallel", leaning_rate=0.1,
                       max_iter=100, tolerance=1e-04, batch_size=NA,
                       random_state=102, centering = FALSE, feature_selection=FALSE,
                       p_value=0.01, rho=0.1, C=0.1, iselasticnet=FALSE){...}
```

This function takes into account several aspects:
- sequential execution with **mode_compute="sequentiel"**
- parallel execution with **ncores=NA, mode_compute="parallel"**
- Execution in Batch, Mini Batch and Online modes with **batch_size=NA**
- Centering reduction of explanatory variables with **centering = FALSE**
- Selection of variables by playing on the arguments **feature_selection=FALSE, p_value=0.01**
- Elasticnet (**Ridge** for **rho**=0 and **Lasso** for **rho**=1) avec les arguments **C** et **rho**.
For each algorithm the principle is explained in the report.

###### sequential execution:

For a sequential execution, specify **comput_mode ='sequentiel'**. <br/>

- BATCH Mode 

```sh
model_batch_seq <- dgrglm.fit(y~., data = data, mode_compute="sequentiel", 
                   leaning_rate=0.1, max_iter=1000,tolerance=1e-06)
summary(model_batch_seq)
```
<br/>

We have overloaded the **print** and **summary** methods for a display adapted to our objects returned by **fit**.

<div align="center">
  <img width="429" alt="Capture d’écran 2021-11-29 à 16 57 01" src="https://user-images.githubusercontent.com/31353252/143910912-c1033fa8-c447-4452-9667-3c7932eb34f2.png">
</div>

- MINI BATCH MODE 

```sh
model_minibatch_seq <- dgrglm.fit(y~., data = data, mode_compute="sequentiel",
                       leaning_rate=0.1, max_iter=2000,tolerance=1e-06,batch_size = 10)
summary(model_minibatch_seq)
```
<br/>
<div align="center">
<img width="436" alt="Capture d’écran 2021-11-29 à 16 56 52" src="https://user-images.githubusercontent.com/31353252/143910933-758993f1-7073-4bcf-894a-ffa2f9f6943c.png">
</div>

- ONLINE MODE 
<br/>
```sh
model_online_seq <- dgrglm.fit(y~., data = data, mode_compute="sequentiel",leaning_rate=0.1, 
                               max_iter=1000,tolerance=1e-06, batch_size = 1)
```
<br/>
<div align="center">
<img width="447" alt="Capture d’écran 2021-11-29 à 17 23 34" src="https://user-images.githubusercontent.com/31353252/143915039-fa61b75c-68e3-4b57-991f-c6056f93357a.png">
</div>

In order to test our different models, we have developed an external function, which displays the ROC curves according to the probabilities of each model. This function is in the trace file devtools_history.R

```sh
perf <- compare_model(probas_mod1=model_batch_seq$probas,
                      probas_mod2=model_minibatch_seq$probas,
                      y=model_minibatch_seq$y_val[,1])
```
<div align="center">
<img width="1279" alt="Capture d’écran 2021-11-29 à 17 06 18" src="https://user-images.githubusercontent.com/31353252/143912390-0fb07683-e8dc-4d2c-845a-f03821bf73c8.png">
</div>
In terms of their ROC curves, the two models are roughly similar in terms of predictions. In terms of their ROC curves, the two models are roughly similar in terms of predictions. Nevertheless model 2 is better.

```sh
perf <- compare_model(probas_mod1=model_batch_seq$probas,
                      probas_mod2=model_online_seq$probas,
                      y=model_online_seq$y_val[,1])
```
<div align="center">
<img width="1280" alt="Capture d’écran 2021-11-29 à 17 36 04" src="https://user-images.githubusercontent.com/31353252/143916401-4c9f134d-4b78-4c34-8522-a0b393fee4fe.png">
</div><br/>
Here we see that the Batch model is clearly better than the online model in terms of prediction


###### parallel execution:

The idea of parallel execution is to slice the data according to the number of cores of the machine and to distribute the calculations on these cores. If the user provides a number of cores not available, the program automatically chooses the max-1 cores.

For a parallel execution, specify **comput_mode ='parallel'** and **nbcores=max-1** in your computer. <br/>

- Mode BATCH

```sh
model_batch_parallel <- dgrglm.fit(y~., data = data, ncores=3, mode_compute="parallel",
                                   leaning_rate=0.1, max_iter=1000,tolerance=1e-06)
```
<div align="center">
<img width="435" alt="Capture d’écran 2021-11-29 à 17 56 28" src="https://user-images.githubusercontent.com/31353252/143918871-240b6411-43b7-48fe-97ea-4c96d531cc57.png">
</div>

- MODE MINI BATCH
```sh

model_minibatch_parallel <- dgrglm.fit(y~., data = data, ncores=3, mode_compute="parallel",
                                       leaning_rate=0.1, max_iter=1000,tolerance=1e-06,
                                       batch_size = 10)
```
   
<div align="center">
<img width="423" alt="Capture d’écran 2021-11-29 à 20 45 30" src="https://user-images.githubusercontent.com/31353252/143940909-8c1747ed-abf1-4131-b46d-9c99262e6fb5.png">
</div>

- MODE ONLINE

```sh
model_online_parallel <- dgrglm.fit(y~., data = data, ncores=3, mode_compute="parallel",
                                    leaning_rate=0.1, max_iter=1000,tolerance=1e-06,
                                    batch_size = 1)

```
   
<div align="center">
<img width="430" alt="Capture d’écran 2021-11-29 à 20 00 22" src="https://user-images.githubusercontent.com/31353252/143935316-66356663-d017-43d6-a2fb-d64150d7af09.png">
</div>



```sh
perf <- compare_model(probas_mod1=model_batch_parallel$probas,
                      probas_mod2=model_minibatch_parallel$probas,
                      y=model_minibatch_parallel$y_val[,1])
                      
perf <- compare_model(probas_mod1=model_batch_parallel$probas,
                      probas_mod2=model_online_parallel$probas,
                      y=model_online_parallel$y_val[,1])
                      
perf <- compare_model(probas_mod1=model_minibatch_parallel$probas,
                      probas_mod2=model_online_parallel$probas,
                      y=model_online_parallel$y_val[,1])                                      
```

<div>
  <img width="1280" alt="1" src="https://user-images.githubusercontent.com/31353252/143941004-2a5c1f76-550e-4dc9-8013-3a7ea91e7abf.png">
  <img width="1280" alt="2" src="https://user-images.githubusercontent.com/31353252/143941017-35fe5f06-8a72-4915-b891-46417aa0734a.png">
  <img width="1280" alt="Capture d’écran 2021-11-29 à 20 53 59" src="https://user-images.githubusercontent.com/31353252/143941527-09d17943-a01d-4a63-9d5c-0c529d9d1cfa.png">
</div>

Looking at the three ROC curves, we can assume that the 03 models (Batch, Mini Batch, Online) are similar in terms of predictions. However there is a big difference on the execution time (see Microbenchmark).


#### Microbenchmark

- **Sequential**

```sh
library(microbenchmark)
microbenchmark(
  model_batch_seq <- dgrglm.fit(y~., data = data, mode_compute="sequentiel",
                                leaning_rate=0.1, max_iter=1000,tolerance=1e-06),
  model_minibatch_seq <- dgrglm.fit(y~., data = data, mode_compute="sequentiel",leaning_rate=0.1,
                                    max_iter=1000,tolerance=1e-06,batch_size = 10),
  model_online_seq <- dgrglm.fit(y~., data = data, mode_compute="sequentiel",
                                 leaning_rate=0.1, max_iter=1000,tolerance=1e-06, batch_size = 1),
  times = 1,
  unit = "s"
)
```
<br/>

<div align="center">
<img width="502" alt="Capture d’écran 2021-11-29 à 20 43 59" src="https://user-images.githubusercontent.com/31353252/143940936-e32bcaa2-bfc5-4142-b41b-1f963db6a398.png">
</div>
<br/>
- **Parallel**
<br/>
```sh
microbenchmark(
  model_batch_parallel <- dgrglm.fit(y~., data = data, ncores=3, mode_compute="parallel",
                                     leaning_rate=0.1, max_iter=1000,tolerance=1e-06),
  model_minibatch_parallel <- dgrglm.fit(y~., data = data, ncores=3, mode_compute="parallel",
                                         leaning_rate=0.1, max_iter=1000,tolerance=1e-06,batch_size = 10),
  model_online_parallel <- dgrglm.fit(y~., data = data, ncores=3, mode_compute="parallel",
                                      leaning_rate=0.1, max_iter=1000,tolerance=1e-06,batch_size = 1),
  times = 1,
  unit = "s"
)
```
<br/>

<div align="center">
<img width="490" alt="Capture d’écran 2021-11-29 à 20 56 33" src="https://user-images.githubusercontent.com/31353252/143941845-337075c7-64a3-46cf-9c5b-91c1877ea479.png">
</div>
<br/>
We implemented logistic regression in sequential and parallel mode. The idea was to see if using all the computational cores a machine has for the execution of the algorithm could increase the speed or not.
According to the results of the Benchmark, there does not seem to be a global gain by executing in parallel (cutting data in lines) as opposed to a sequential execution

To verify this hypothesis we will generate logistic data manually and play on the number of individuals and variables and test for each case the computation time.
<br/>

###### Logistic data generation
```sh
set.seed(103)
n <-1000000  # Number of obervations
p <- 5 # Number of variables
theta = runif(p+1) # Theta vector
X <- cbind(1,matrix(rnorm(n*p),n,p))
Z <- X %*% theta # linear combination of variables
fprob <- ifelse(Z<0, exp(Z)/(1+exp(Z)),1/(1+exp(-Z))) # Calcul des probas d'affectation
y<- rbinom(n,1,fprob)
data = as.data.frame(cbind(X,y))
data$V1 <- NULL # delete colomn de biais. It will be created when fit is called
```
The dataset is well generated. It contains 1 million rows<br/>

<div align="center">
<img width="511" alt="Capture d’écran 2021-11-29 à 21 17 44" src="https://user-images.githubusercontent.com/31353252/143944384-21b44e31-8da6-4935-97cf-fca8568056a7.png">
</div>
<br/>
In the following we will just play on n and p to increase or decrease the number of variables and/or individuals. Now, Let's run the Microbenchmark again on the set of 1 million rows and 5 variables.

<br/>
- **Sequential**
<br/>
<div align="center">

</div>

- **Parallel**
<br/>

<div align="center">

</div>
<br/>

#### Features Selection
<br/>
We have the possibility to create the model on variables selected automatically according to their relevance.<br/>

```sh
 model_batch_parallel <- dgrglm.fit(y~., data = data, ncores=3, mode_compute="parallel",
                                    leaning_rate=0.1, max_iter=1000,tolerance=1e-06, 
                                    feature_selection=TRUE, p_value=0.01)
```

<br/>

#### Centering reduction

  - centering = FALSE
<br/>

```sh
model_batch_parallel <- dgrglm.fit(y~., data = data, ncores=3, mode_compute="parallel",
                                  leaning_rate=0.1,max_iter=1000,tolerance=1e-06, 
                                  feature_selection=TRUE, p_value=0.01,
                                  centering = FALSE)
                                     
```
<br/>

#### Elasticnet

  - **iselasticnet=TRUE, rho=0.1, C=0.1**
 
```sh
model_batch_parallel <- dgrglm.fit(y~., data = data, ncores=3, mode_compute="parallel",leaning_rate=0.1,
                                   max_iter=1000, tolerance=1e-06, 
                                   feature_selection=TRUE, p_value=0.01,
                                   centering = FALSE, 
                                   iselasticnet=TRUE, rho=0.1, C=0.1)                                   
```

- si **rho=0 ===> RIDGE**
- si **rho=1 ===> LASSO**

<br/>

### 2. Multinomiale regression

- For multiclass regression the target variable is coded in binary (One vs ALL).

```sh
dgrglm.multiclass.fit(formule, data, leaning_rate=0.1, max_iter=3000, tolerance=1e-04, 
                      random_state=102, centering = FALSE){...}                                   
```



