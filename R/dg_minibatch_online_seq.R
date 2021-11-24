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

  # Controle du max iteratons
  if (max_iter <= 0){
    stop("'max_iter' must be greater than zero")
  }
  # Controle de dimension
  if (dim(X)[1] != length(y)){
    stop("the dimensions of 'x' and 'y' do not match")
  }
  # Initialiser le generateur de nombre aleatoire pour rendre reproductible les calculs
  if(!is.na(random_state)){
    set.seed(random_state)
  }
  xy = cbind(X,y)
  #remove NA rows
  X <- na.omit(X)
  y <- na.omit(y)
  # Vecteur de cout
  cost_vector = c()
  m = nrow(X)
  # controle de convergence
  converge = FALSE
  iter <- 0
  nb_iter_ <- 0
  instance <- list()
  while((iter < max_iter) && (converge == FALSE) ){
    #iteration suivante
    iter <- iter + 1
    # SHUFLE the dataset
    rows <- sample(nrow(xy))  # Melanger les indices du dataframe xy
    xy <- xy[rows, ] # Utiliser ces indices pour reorganiser le dataframe
    # MINI BATCH
    for (start in seq(from=1, to=dim(X)[1], batch_size)){
      stop = start + batch_size
      if(stop > dim(X)[1]){
        break
      }
      xBatch = xy[start:stop,-ncol(xy)]
      yBatch = xy[start:stop, ncol(xy)]
      # Calcul du cout
      cost = logLoss(theta, xBatch, yBatch)
      # Historisation de la fonction de cout
      cost_vector = c(cost_vector, cost)
      # conteur permettant de suivre le nomre d'element de history
      nb_iter_ = nb_iter_ +1
      # Mise à jour du theta
      grd = gradient(theta, xBatch, yBatch)
      new_theta = theta - leaning_rate*grd
      if (sum(abs(new_theta - theta)) < tolerance){
        converge <- TRUE
        break
      }
      theta = new_theta
    }
  }
  instance$theta <- theta
  instance$history_cost <- cost_vector
  instance$nb_iter_while <- iter
  instance$nb_iter_for <-nb_iter_
  class(instance) <- "global_gradient_descent"
  return(instance)
}
