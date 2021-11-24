#' MiniBatch DGSRow Distributed
#'
#' @param X is the matrix of our predictor variables with the bias column
#' @param y is a target variable to predict
#' @param theta is a vector containing the parameters or coefficient of the logistic to be estimated
#' @param ncores parameters representing the number of cores to be used for parallel execution
#' @param batch_size a parameter that specifies the number of observations in each mini-batch. It can significantly affect performance
#' @param leaning_rate is the learning rate that controls the magnitude of the vector update.
#' @param max_iter is the number of iterations.
#' @param tolerance an additional parameter which specifies the minimum movement allowed for each iteration
#'
#' @import parallel
#'
#' @return this function returns an instance containing:
#' \itemize{
#'  \item final theta
#'  \item history cost
#'  \item iteration number
#'  \item iteration number minibatch
#' }
#' @export
#'
#' @examples
#' \dontrun{
#'  dgsrow_minibatch_parallle_bis(X,y,theta)
#'  dgsrow_minibatch_parallle_bis(X,y,theta,ncores=3)
#' }
dgsrow_minibatch_parallle2<- function(X,y,theta, batch_size, ncores, leaning_rate, max_iter, tolerance){
  # Controle de dimension
  if (dim(X)[1] != length(y)){
    stop("les dimensions de 'x' et 'y' ne correspondent pas")
  }

  iter_while <- 0
  iter_for <- 0
  cost_vector = c()
  # Paralellisation du calcul de gradient
  cl <- makeCluster(ncores)
  clusterExport(cl, c("theta","sigmoid","gradient"))
  start <- 1
  converge <- FALSE
  while((iter_while < max_iter) && (converge==FALSE)){
    iter_while <- iter_while + 1
    task <- function(k){
      # Sample group for each node
      app_X<- data_app[data_app$fold == k, -1]
      app_Y <- data_app[data_app$fold == k, 1]
      # delete colonne fold
      app_X$fold = NULL
      app_Y$fold = NULL
      grad <- gradient(theta,app_X,as.integer(app_Y))
      return(grad)
    }
    for (start in seq(from=1, to=dim(X)[1], batch_size)){
      iter_for<- iter_for + 1
      stop = start + (batch_size-1)
      if(stop >= dim(X)[1]){
        stop = dim(X)[1]
        break
      }
      xBatch = X[start:stop,]
      yBatch = y[start:stop]
      xBatch$biais <-  1
      df <- cbind(yBatch,xBatch)
      data_app = decoupage_ligne(df, ncores)
      res <- clusterApply(cl, x=1:ncores, task)
      gradient_aggr <- apply(sapply(res, function(x) x),1,sum)
      new_theta <- theta - (leaning_rate*gradient_aggr)/batch_size
      if (sum(abs(theta-new_theta)) < tolerance){
        converge =TRUE
        break
      }
      theta <- new_theta
      cost = logLoss(theta, xBatch, yBatch)
      cost_vector = c(cost_vector, cost)
      clusterExport(cl, "theta")
    }
  }
  stopCluster(cl)
  return(list(theta_final = theta, history_cost = cost_vector, nbIter=iter_while, nbIter_for = iter_for))
}
