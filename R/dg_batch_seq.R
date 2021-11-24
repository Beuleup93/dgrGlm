#' Gradient descent algorithm
#'
#' @param X is the matrix of our predictor variables with the bias column
#' @param y is a target variable to predict.
#' @param theta is a vector containing the parameters or coefficient of the logistic to be estimated
#' @param leaning_rate is the learning rate that controls the magnitude of the vector update.
#' @param max_iter is the number of iterations.
#' @param tolerance an additional parameter which specifies the minimum movement allowed for each iteration
#' @importFrom stats na.omit
#'
#' @export
#' @author "Saliou NDAO <salioundao21@gmail.com>"
#' @return this function returns the instance of model with all parameters
#'
#' @examples
#' \dontrun{
#'   gradient(X,y,theta)
#'   gradient(X,y,theta, leaning_rate=0.1, max_iter=100, tolerance=1e-04)
#' }
dg_batch_seq<- function(X,y,theta, leaning_rate, max_iter, tolerance){
  if (dim(X)[1] != length(y)){
    stop("the dimensions of 'x' and 'y' do not match")
  }
  X <- na.omit(X)
  y <- na.omit(y)

  cost_vector = c()
  m = nrow(X)
  converge = FALSE
  iter <- 0
  instance <- list()
  while((iter < max_iter) && (converge == FALSE) ){
    iter <- iter +1
    PI <- sigmoid(X%*%theta)
    cost = logLoss(theta, X, y)
    cost_vector = c(cost_vector, cost)
    gradient = gradient(theta, X, y)
    new_theta = theta - leaning_rate*gradient
    if (sum(abs(new_theta-theta)) < tolerance){
      converge <- TRUE
    }
    theta = new_theta
  }
  instance$theta <- theta
  instance$history_cost <- cost_vector
  instance$nb_iter_while <- iter
  class(instance) <- "gradient_desc_batch"
  return(instance)
}
