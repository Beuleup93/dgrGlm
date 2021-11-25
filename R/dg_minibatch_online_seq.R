#' Global Gradient descent algorithm
#'
#' This function allows the execution of the binary logistic regression according to the batch, mini batch and online mode.
#'
#' @param X is the matrix of our predictor variables with the bias column
#' @param y is a target variable to predict.
#' @param theta is a vector containing the parameters or coefficient of the logistic to be estimated
#' @param batch_size a parameter that specifies the number of observations in each mini-batch. It can significantly affect performance
#' @param random_state this parameter defines the seed of the random number generator, use when shuffling to mix observations.
#' @param leaning_rate is the learning rate that controls the magnitude of the vector update.
#' @param max_iter is the number of iterations.
#' @param tolerance an additional parameter which specifies the minimum movement allowed for each iteration
#'
#' @importFrom stats na.omit
#' @export
#' @return this function returns an instance containing:
#' \itemize{
#'  \item final theta
#'  \item history cost
#'  \item iteration number
#'  \item intern iteration number
#' }
#'
#' @examples
#' \dontrun{
#'  global_grad_descent(X,y,theta)
#' }
dg_batch_minibatch_online_seq<- function(X,y,theta, batch_size, random_state, leaning_rate, max_iter, tolerance){
  instance <- list()

  # CONTROL OF MATCH DIMENSION
  if (dim(X)[1] != length(y)){
    stop("the dimensions of 'x' and 'y' do not match")
  }
  yx = as.data.frame(cbind(y,X))

  # INIT COST HISTORISATION AND ITERATIONS
  cost_vector = c()
  iter <- 0
  nb_iter_ <- 0

  while(iter < max_iter ){
    iter <- iter + 1

    # BATCH, MINI BATCH OR ONLINE
    for (start in seq(from=1, to=dim(X)[1], batch_size)){
      stop = start + (batch_size-1)
      if(stop > dim(X)[1]){
        break
      }
      # DATA FOR MINI BATCH IN ITERIATION I
      xBatch = yx[start:stop,-1]
      yBatch = yx[start:stop, 1]

      # HISTORY COST
      cost = logLoss(theta, as.matrix(xBatch), yBatch)
      cost_vector = c(cost_vector, cost)
      nb_iter_ = nb_iter_ +1

      # GRADIENT CALCULATION AND CALCUL NEW THETA
      grd = gradient(theta, as.matrix(xBatch), yBatch)
      new_theta = theta - leaning_rate*grd

      # CONTROL OF CONVERGENCE
      if (sum(abs(new_theta - theta)) < tolerance){
        break
      }
      # UPDATE THETA
      theta = new_theta
    }
  }
  #________________________________________
  instance$theta <- theta
  instance$history_cost <- cost_vector
  instance$nb_iter_while <- iter
  instance$nb_iter_for <-nb_iter_
  class(instance) <- "global_gradient_descent"
  return(instance)
}
