#'  Batch DGSRow Distributed
#'
#' @param X is the matrix of our predictor variables with the bias column
#' @param y is a target variable to predict
#' @param theta is a vector containing the parameters or coefficient of the logistic to be estimated
#' @param ncores parameters representing the number of cores to be used for parallel execution
#' @param leaning_rate is the learning rate that controls the magnitude of the vector update
#' @param max_iter is the number of iterations
#' @param tolerance an additional parameter which specifies the minimum movement allowed for each iteration
#' @param rho hyper parameter which allows arbitration between RDIGE and LASSO.
#' @param C parameter allowing to arbitrate between the penalty and the likelihood in the guidance of the modeling.
#' @return this function returns an instance containing:
#' \itemize{
#'  \item final theta
#'  \item history cost
#'  \item iteration number
#' }
#' @export
#' @import parallel
#'
#' @examples
#' \dontrun{
#'  dgsrow_batch_parallele(X,y,theta)
#'  dgsrow_batch_parallele(X,y,theta,ncores=3)
#' }
dgsrow_batch_parallele<- function(X,y,theta, ncores, leaning_rate, max_iter, tolerance, rho=NA, C=NA){
  instance <- list()

  # CONTROL OF DIMENSION
  if (dim(X)[1] != length(y)){
    stop("les dimensions de 'x' et 'y' ne correspondent pas")
  }

  # ONLINE DATA DECOUPAGE
  df <- as.data.frame(cbind(y,X))
  data_group = decoupage_ligne(df,ncores)

  # FUNCTION PERFORMED BY EACH CLUSTER
  task <- function(k){
    # DECOUPAGE DATA FOR EACH CORE
    train_X<- data_group[data_group$fold == k, -1]
    train_Y <- data_group[data_group$fold == k, 1]

    # DELETE COLUMN FOLD
    train_X$fold = NULL
    train_Y$fold = NULL

    # CALCUL GRADIENT
    if(is.na(C) && is.na(rho)){
      grad <- gradient(theta,as.matrix(train_X),as.integer(train_Y))
    }else{
      grad <- gradientElasticnet(theta,as.matrix(train_X),as.integer(train_Y), rho,C)
    }
    return(grad)
  }

  # CLUSTER INSTANCIATION AND OBJECTS EXPORT
  cl <- makeCluster(ncores)
  clusterExport(cl, c("theta", "sigmoid","gradient"),envir=environment())

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
    new_theta <- theta - (leaning_rate*gradient_glob)

    # CONTROL CONVERGENCE
    if (sum(abs(theta-new_theta)) < tolerance){
      break
    }

    # UPDATE THETA
    theta = new_theta

    # COST CALCULATION
    if(is.na(C) && is.na(rho)){
      cost = logLoss(theta, as.matrix(X), y)
    }else{
      cost = logLossElasticnet(theta, as.matrix(X), y, rho, C)
    }

    # HISTORIZATION OF THE COST FUNCTION
    cost_vector = c(cost_vector, cost)

    # SEND NEW THETA TO CORE OR CLUSTER
    clusterExport(cl, "theta",envir=environment())
  }
  stopCluster(cl)
  #___________________________________
  instance$theta <- theta
  instance$history_cost<- cost_vector
  instance$nbIter <- iter
  class(instance) <- "GDS_BATCH_PARALEL"

  return(instance)
}
