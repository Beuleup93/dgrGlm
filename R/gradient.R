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
