#' Logistic regression cost function
#'
#' The cost function, or loss function is the function to be minimized(or maximized) by varying the decision variables.
#'
#' @param theta is a vector containing the parameters or coefficient of the logistic to be estimated
#' @param X is the matrix of our predictor variables with the bias column
#' @param y is a target variable to predict
#'
#' @author "Saliou NDAO <salioundao21@gmail.com>"
#' @export
#' @return this function returns a scalar corresponding to the cost for this vector of parameters \code{theta}
#'
#' @examples
#' \dontrun{
#'   log_loss(theta, X, y)
#' }
logLoss <- function(theta, X, y){
  n <- length(y)
  PI <- sigmoid(X %*% theta)
  J <- (t(-y)%*%log(PI)-t(1-y)%*%log(1-PI))/n
  return(J)
}


#' Title
#'
#' @param theta is a vector containing the parameters or coefficient of the logistic to be estimated
#' @param X is the matrix of our predictor variables with the bias column
#' @param y is a target variable to predict
#' @param rho hyper parameter which allows arbitration between RDIGE and LASSO
#' @param C parameter allowing to arbitrate between the penalty and the likelihood in the guidance of the modeling
#'
#' @return a scalar of the penalized cost function
#' @export
#'
#' @examples
#' \dontrun{
#'   logLossElasticnet(theta, X, y,l1,l2)
#' }
logLossElasticnet <- function(theta, X, y, rho, C){
  n <- length(y)
  Z = X %*% theta
  J <- C*(sum(log(exp(-y*(Z))+1)))+rho*sum(abs(theta)) + ((1-rho)/2)*sum(theta^2)
  return(J)
}





