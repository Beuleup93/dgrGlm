#'  Batch DGSRow Distributed
#'
#' @param X is the matrix of our predictor variables with the bias column
#' @param y is a target variable to predict
#' @param theta is a vector containing the parameters or coefficient of the logistic to be estimated
#' @param ncores parameters representing the number of cores to be used for parallel execution
#' @param leaning_rate is the learning rate that controls the magnitude of the vector update
#' @param max_iter is the number of iterations
#' @param tolerance an additional parameter which specifies the minimum movement allowed for each iteration
#'
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
dgsrow_batch_parallele<- function(X,y,theta, ncores, leaning_rate, max_iter, tolerance){
  # Controle de dimension
  if (dim(X)[1] != length(y)){
    stop("les dimensions de 'x' et 'y' ne correspondent pas")
  }
  df <- cbind(y,X)
  # Decoupage données en ligne
  data_app = decoupage_ligne(df, (detectCores()-1))
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
  # Instanciation des cluster et export des objets
  cl <- makeCluster(ncores)
  clusterExport(cl, c("theta", "sigmoid","gradient"))
  iter <- 0
  cost_vector = c()
  while(iter < max_iter){
    iter <- iter + 1
    # Appel de clusterApply pour paralleliser le calcule des gradients
    res <- clusterApply(cl, x=1:ncores, task)
    # Aggrégation des gradient
    gradient_glob = apply(sapply(res, function(x) x),1,sum)
    # Mise à jour du theta par le master
    new_theta <- theta - (leaning_rate*gradient_glob)
    # Controle de convergence
    if (sum(abs(theta-new_theta)) < tolerance){
      break
    }
    theta = new_theta
    # Calcul du cout
    cost = logLoss(theta, X, y)
    # Historisation de la fonction de cout
    cost_vector = c(cost_vector, cost)
    # Envoit du nouveau theta au master
    clusterExport(cl, "theta")
  }
  stopCluster(cl)
  return(list(theta_final = theta, history_cost = cost_vector, nbIter=iter))
}
