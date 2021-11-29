#' Customization function of the print method for the model instance
#'
#' @param x model instance returned by dgrglm.fit function
#' @param ... other params
#'
#' @import utils
#'
#' @export
print.modele <- function(x, ...){
  modele<-x
  cat("Coefficents:\n",c("(intercept)",modele$explicatives),"\n",modele$res$theta,"\n\n")
  cat("number of iterations: ",modele$res$nbIter,"\n")
  cat("error rate: ",modele$err_rate,"\n")
  cat("Null Deviance: ",modele$metric$nulldev," on ",nrow(modele$probas)-1," degrees of freedom","\n")
  cat("Residual Deviance: ",modele$metric$resdev," on ",nrow(modele$probas)-length(modele$explicatives)-1," degrees of freedom","\n")
  cat("AIC: ",modele$metric$aic)
}




