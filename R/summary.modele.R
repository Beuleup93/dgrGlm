#' Customization function of the summary method for the modele instance
#'
#' @param object model instance returned by dgrglm.fit function
#' @param ... other argument
#'
#' @importFrom stats addmargins
#' @export
#'
summary.modele <- function(object, ...){
  modele<- object
  cm <- table(modele$y_val$TrueY,modele$y_val$Ypredict)
  names(dimnames(cm)) <- c("observed","predicted")
  cm <- addmargins(cm)
  error_rate <- ((cm[2,1]+cm[1,2])/cm[3,3])*100
  accuracy <- 100-error_rate
  recall <- (cm[2,2]/(cm[2,2]+cm[2,1]))*100
  false_positive_rate <- (cm[1,2]/cm[1,3])*100
  true_negative_rate <- (cm[1,1]/cm[1,3])*100
  precision <- (cm[2,2]/cm[3,2])*100
  f1_score <- (2*precision*recall)/(precision+recall)
  cat("Confusion matrix:\n\n")
  print(cm)
  cat("\n\n")
  cat("Error rate : ",error_rate,"%\n")
  cat("Accuracy : ",accuracy,"%\n")
  cat("Precision : ",precision,"%\n")
  cat("Recall : ",recall,"%\n")
  cat("F1-score : ",f1_score,"%\n")
  cat("False positive rate :",false_positive_rate,"%\n")
  cat("True negative rate :",true_negative_rate,"%\n\n")
  cat("Log-likelihood model: ",modele$metric$LLmodel,"\n")
  cat("Log-likelihood null model: ",modele$metric$LLnull,"\n")
  cat("Null Deviance: ",modele$metric$nulldev," on ",nrow(modele$probas)-1," degrees of freedom","\n")
  cat("Residual Deviance: ",modele$metric$resdev," on ",nrow(modele$probas)-length(modele$explicatives)-1," degrees of freedom","\n")
  cat("AIC: ",modele$metric$aic)
}
