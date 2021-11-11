#' Function fit to construct model
#'
#' This function allows us to create a binary logistic regression model
#'
#' @param formule allows you to define the target variable and predictor variables
#' @param data the data source containing all the variables specified in the formula
#' @param mode The mode of updating the coefficients of the model (BATCH(default), MINI_BATCH and ONLINE)
#' @param leaning_rate is the learning rate that controls the magnitude of the vector update.
#' @param max_iter is the number of iterations.
#' @param tolerance an additional parameter which specifies the minimum movement allowed for each iteration
#' @param batch_size a parameter that specifies the number of observations in each mini-batch. It can significantly affect performance
#' @param random_state this parameter defines the seed of the random number generator, use when shuffling to mix observations.
#' @import plyr
#' @importFrom stats formula as.formula runif model.frame
#'
#' @return an instance of model
#' @export
#'
#' @examples
#' \dontrun{
#'  fit(formule, data)
#' }
fit <- function(formule, data, mode="BATCH", leaning_rate=0.1, max_iter=100, tolerance=1e-04, batch_size=NA, random_state=1){

  if(!is.formula(formule)){
    stop("formula must be of type formula")
  }

  if(!is.data.frame(data)){
    stop("The data source must be a data frame")
  }
  f = formula(formule)
  colonne_names = colnames(data)
  for (v in all.vars(f)){
    if(!is.element(v, colonne_names) && v != '.'){
      print(paste("Whoops!! Correspondence error between variables:: -->", v))
      stop("Check the concordance between the columns of the formula and those of the data source")
    }
  }
  # Recuperer les colonnes du bloc de données qui correspondent aux arguments du formula
  df = model.frame(formula = as.formula(formule), data = data)
  X = df[,-1]
  X$biais = 1
  y = df[,1]
  # Initialisation des parametres du modele
  theta = runif(ncol(X))
  X = as.matrix(X)
  y = as.vector(y)
  instance <- list()
  if(mode=="BATCH"){
    instance$res <- grad_descent_batch(X,y,theta,leaning_rate=leaning_rate, max_iter=max_iter, tolerance=tolerance)
  } else if(mode == "MINI_BATCH"){
    instance$res <- global_grad_descent(X,y,theta, batch_size=batch_size, random_state=random_state, leaning_rate=leaning_rate, max_iter=max_iter, tolerance=tolerance)
  }else{
    instance$res <- global_grad_descent(X,y,theta, batch_size=batch_size, random_state=random_state, leaning_rate=leaning_rate, max_iter=max_iter, tolerance=tolerance)
  }
  class(instance) <- "modele"
  return(instance)
}
