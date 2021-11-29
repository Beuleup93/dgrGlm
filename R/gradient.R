#' The gradient of the objective function
#'
#' The gradient function represents the partial derivative of the cost function with respect to each coefficient of the model
#'
#' @param theta is a vector containing the parameters or coefficient of the logistic to be estimated
#' @param X is the matrix of our predictor variables with the bias column
#' @param y is a target variable to predict
#'
#' @author "Saliou NDAO <salioundao21@gmail.com>"
#' @export
#' @return this function returns a gradient vector
#'
#' @examples
#' \dontrun{
#'   gradient(theta, X, y)
#' }
gradient <- function(theta, X, y){
  n <- length(y)
  PI <- sigmoid(X%*%theta)
  gradient <- (t(X)%*%(PI - y))/n
  return (as.vector(gradient))
}
#'  Gradient for Elasticnet Loss Function
#'
#' @param theta is a vector containing the parameters or coefficient of the logistic to be estimated
#' @param X is the matrix of our predictor variables with the bias column
#' @param y is a target variable to predict
#' @param rho hyper parameter which allows arbitration between RDIGE and LASSO
#' @param C parameter allowing to arbitrate between the penalty and the likelihood in the guidance of the modeling
#'
#' @return a vector
#' @import numDeriv
#' @export
#'
#' @examples
#' \dontrun{
#'   gradientElasticnet(theta, X, y,l1,l2)
#' }
gradientElasticnet <- function(theta, X, y, rho, C){
  #Z <- X %*% theta
  #dtheta.l1 <- ifelse(theta>0,1,ifelse(theta<0,-1,0))
  #gradEN <- (1-rho)*sum(theta)+sum(dtheta.l1)+C*sum(((y*X)*exp(-y*Z)/exp(-y*Z)+1))
  gradEN = grad(logLossElasticnet, x=theta, X = X,y=y,rho=rho,C=C)
  return (as.vector(gradEN))
}





