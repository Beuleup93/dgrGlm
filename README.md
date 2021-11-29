# dgrGlm: GRadient Descent For Logistic Regression

### DESCRIPTION

This project is part of our training in Data Science at the University of Lyon 2.  The main objective is to develop an R package under S3 that allows to do binary 
logistic regression distributed on the different cores of the user's computer. Gradient descent is used for the optimization of the parameters. 
Here the idea is to allow the user to take advantage of the totality of these computer resources, if of course this can help to make the execution 
of the calculations faster. Here are the different functionalities of our package that we will present in the following lines:

- Binary logistic regression model in **sequential** mode 
- Splitting of calculations and data for **parallel** execution
- Binary logistic regression model in parallel mode
- Automatic Features selection
- **Elasticnet** (Ridge and Lasso)
- Multiple logistic regression 
- Model comparison


### Logistic Regression

Logistic regression is an old and well-known statistical predictive method that applies to binary classification, but can be easily extended to the multiclass 
framework (multinomial regression in particular, but not only...).It has become popular in recent years in machine learning thanks to "new" computational 
algorithms (optimization) and powerful libraries.
The work on regularization makes its use efficient in the context of high dimensional learning (nb. Var. Expl. >> nb. Obsv.). It is related to neural networks 
(simple perceptron).

The objective of logistic regression is to explain and predict the values of a qualitative variable Y, most often binary, from qualitative and qualitative 
explanatory variables X = (X1,...,Xp). If we note 0 and 1 the modalities of Y, the logistic model is written: 
<div align="center">
<img  width="509" alt="Capture d’écran 2021-11-29 à 09 55 17" src="https://user-images.githubusercontent.com/31353252/143846346-295455c7-c0dc-456e-8532-48658735f050.png">
</div>

where P(X) is the probability P(Y=1|X=X) and x=(x1,...,xp) is a realization of X=(X1,...,Xp). The log(u/1+u) function relates the probability p(x) to the linear 
combination of explanatory variables. The coefficients b0,...,bp are estimated by the maximum likelihood method from the observations. 

<div align="center">
<img width="219" alt="Capture d’écran 2021-11-29 à 09 50 50" src="https://user-images.githubusercontent.com/31353252/143847495-71f03f29-efcb-46e2-ac61-984debaa38ad.png">
<img width="459" alt="Capture d’écran 2021-11-29 à 11 54 02" src="https://user-images.githubusercontent.com/31353252/143863890-05eb81ce-4211-4d78-894a-28cc13ab71e4.png">
</div>

With 𝜋, we have a true "score" i.e., a true estimate of the probability of belonging to the target modality (0 ≤ 𝜋 ≤ 1).

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

In order to test our functions, we will work with the dataset **ionosphere.xlsx**. It consists of 351 obervations and 34 variables.

<div align="center">
<img width="490" alt="Capture d’écran 2021-11-29 à 15 59 47" src="https://user-images.githubusercontent.com/31353252/143900971-229feff3-18f8-4cba-89ce-0efcba432037.png">
</div>

### Binary Logistic Regression

We will start by testing the binary logistic regression on our dataset. The variable to be explained is Y and the explanatory variables are a03,...,a34





