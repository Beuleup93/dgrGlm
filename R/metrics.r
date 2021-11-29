#' Metrics Function
#'
#' @param pi linear combination of predictor variables
#' @param y target variable to predict
#' @param p number of model parameters
#'
#' @export
#' @return an S3 instance containing the values for the AIC (Akaike information criterion), the null deviance and the residual deviance
#'
#' @examples
#' \dontrun{
#'   metrics(pi,y,p)
#' }

metrics <- function(pi,y,p){
  pinull <- replicate(length(pi),0.5)
  LLmodel <- sum(log((pi^y)*((1-pi)^(1-y))))
  LLnull <- sum(log((pinull^y)*((1-pinull)^(1-y))))
  instance <- list()
  instance$LLmodel <- LLmodel
  instance$LLnull <- LLnull
  instance$nulldev <- 2*(-(LLnull))
  instance$resdev <- 2*(-(LLmodel))
  instance$aic <- (2*p)-(2*LLmodel)
  class(instance) <- "METRICS"
  return(instance)
}
