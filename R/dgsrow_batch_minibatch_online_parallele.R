#' Batch Mini & Online DGSRow Distributed
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
#' }
#' @export
#'
#' @examples
#' \dontrun{
#'  dgs_minibatch_online_parallle(X,y,theta)
#'  dgs_minibatch_online_parallle(X,y,theta,ncores=3)
#' }
dgs_minibatch_online_parallle<- function(X, y, theta, ncores, batch_size, leaning_rate, max_iter, tolerance){
  # Control of dimension
  if (dim(X)[1] != length(y)){
    stop("les dimensions de 'x' et 'y' ne correspondent pas")
  }
  df <- cbind(y,X)
  # Decoupage ligne de mon jeu
  data_app = decoupage_ligne(df, (detectCores()-1))
  task <- function(k){
    # Sample group for each node
    app_X<- data_app[data_app$fold == k, -1]
    app_Y <- data_app[data_app$fold == k, 1]
    # delete colonne fold
    app_X$fold = NULL
    app_Y$fold = NULL

    # Iteration pour chaque batch_size
    xy <- cbind(as.matrix(app_X),app_Y)
    gradient_aggr <- rep(0,ncol(app_X))
    for (start in seq(from=1, to=dim(app_X)[1], batch_size)){
      stop = start + (batch_size-1)
      if(stop >= dim(app_X)[1]){
        break
      }
      xBatch = app_X[start:stop,]
      yBatch = app_Y[start:stop]
      # aggregate gradient
      gradient_aggr <- gradient_aggr+gradient(theta,xBatch,as.integer(yBatch))
    }
    return(gradient_aggr)
  }
  # Distributed compute gradient
  cl <- makeCluster(ncores)
  clusterExport(cl, c("theta","sigmoid","gradient"))
  iter <- 0
  cost_vector = c()
  while(iter < max_iter){
    iter <- iter + 1
    res <- clusterApply(cl, x=1:ncores, task)
    gradient_glob = apply(sapply(res, function(x) x),1,sum)
    new_theta <- theta - (leaning_rate*gradient_glob)/batch_size
    if (sum(abs(theta-new_theta)) < tolerance){
      break
    }
    theta = new_theta
    cost = logLoss(theta, X, y)
    cost_vector = c(cost_vector, cost)
    clusterExport(cl, "theta")
  }
  stopCluster(cl)
  return(list(theta_final = theta, history_cost = cost_vector, nbIter=iter))
}
