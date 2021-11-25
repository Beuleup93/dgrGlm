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
  instance <- list()

  # CONTROL OF DIMENSION
  if (dim(X)[1] != length(y)){
    stop("les dimensions de 'x' et 'y' ne correspondent pas")
  }

  # ONLINE DATA DECOUPAGE
  df <- as.data.frame(cbind(y,X))
  data_group = decoupage_ligne(df, ncores)

  # FUNCTION PERFORMED BY EACH CLUSTER
  task <- function(k){
    # DECOUPAGE DATA FOR EACH CORE
    app_X <- data_group[data_group$fold == k, -1]
    app_Y <- data_group[data_group$fold == k, 1]

    # DELETE COLUMN FOLD
    app_X$fold = NULL
    app_Y$fold = NULL

    # INIT GRADIENT_AGGR FOR AGGREGATION
    gradient_aggr <- rep(0,ncol(app_X))

    # DO THIS FOR EVERY BATCH IN CORE CALCULTATION
    for (start in seq(from=1, to=dim(app_X)[1], batch_size)){
      stop = start + (batch_size-1)
      if(stop > dim(app_X)[1]){
        break
      }

      # DATA FOR MINI BATCH IN ITERIATION I
      xBatch = app_X[start:stop,]
      yBatch = app_Y[start:stop]

      # GRADIENT CALCULATION AND AGREGATION
      gradient_aggr <- gradient_aggr+gradient(theta,as.matrix(xBatch),as.integer(yBatch))
    }
    return(gradient_aggr)
  }
  # CLUSTER INSTANCIATION AND OBJECTS EXPORT
  cl <- makeCluster(ncores)
  clusterExport(cl, c("theta","sigmoid","gradient"))

  # INIT HISTORY COST AND ITERATION
  iter <- 0
  cost_vector = c()

  while(iter < max_iter){
    iter <- iter + 1
    # FOR PARALLELIZE CALCULATION OF GRADIENTS
    res <- clusterApply(cl, x=1:ncores, task)

    # AGGREGATES GRADIENTS OF ALL CLUSTERS
    gradient_glob = apply(sapply(res, function(x) x),1,sum)

    # NEW THETA CALCULATION
    new_theta <- theta - (leaning_rate*gradient_glob)/batch_size

    # CONTROL OF CONVERGENCE
    if (sum(abs(theta-new_theta)) < tolerance){
      break
    }
    # UPDATE OLD THETA
    theta = new_theta
    # CALCUL AND HISTORIZATION OF THE COST FUNCTION
    cost = logLoss(theta, as.matrix(X), y)
    cost_vector = c(cost_vector, cost)

    # SEND THETA TO CORE OR CLUSTER FOR OTHER GRADIENT CALCULATION
    clusterExport(cl, "theta")
  }
  stopCluster(cl)
  #___________
  instance$theta <- theta
  instance$history_cost<- cost_vector
  instance$nbIter <- iter
  class(instance) <- "GDS_MINIBATCH_PARALEL"
  return(instance)
}
