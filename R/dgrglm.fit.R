#' Function fit to construct model
#'
#' This function allows us to create a binary logistic regression model
#'
#' @param formule allows you to define the target variable and predictor variables
#' @param data the data source containing all the variables specified in the formula
#' @param ncores parameters representing the number of cores to be used for parallel execution
#' @param mode_compute algorithm mode of execution. thera are "parallel" and "sequentiel"
#' @param leaning_rate is the learning rate that controls the magnitude of the vector update
#' @param max_iter is the number of iterations
#' @param tolerance an additional parameter which specifies the minimum movement allowed for each iteration
#' @param batch_size a parameter that specifies the number of observations in each mini-batch. It can significantly affect performance
#' @param random_state this parameter defines the seed of the random number generator, use when shuffling to mix observations
#' @param centering to center and reduce the variables, by default FALSE
#' @import plyr
#' @import parallel
#' @importFrom stats formula as.formula runif model.frame
#'
#' @return an instance of model
#' @export
#'
#' @examples
#' \dontrun{
#'  dgrglm.fit(formule, data)
#'  dgrglm.fit(formule, data,ncores=3, mode_compute="parallel",)
#' }
dgrglm.fit <- function(formule, data, ncores=3, mode_compute="parallel", leaning_rate=0.1, max_iter=100, tolerance=1e-04, batch_size=NA, random_state=1, centering = FALSE){

  # Objet S3
  instance <- list()

  # Controle de saisie utilisateur
  if(!is.formula(formule)){
    stop("formula must be of type formula")
  }

  if(!is.data.frame(data)){
    stop("The data source must be a data frame")
  }

  if (leaning_rate <= 0){
    stop("'learn_rate' must be greater than zero")
  }

  if (tolerance <= 0){
    stop("'tolerance' must be greater than zero")
  }

  if (max_iter <= 0){
    stop("'max_iter' must be greater than zero")
  }

  if( (batch_size <=  0) || (batch_size > dim(X)[1]-1)){
    stop("'Batch size' must be between 1 and nbObs-1 ")
  }

  if(ncores<=0 || ncores>=detectCores()){
    ncores = detectCores()-1
  }

  # Controle de correspondance des colonnes
  f = formula(formule)
  colonne_names = colnames(data)
  for (v in all.vars(f)){
    if(!is.element(v, colonne_names) && v != '.'){
      print(paste("Whoops!! Correspondence error between variables:: -->", v))
      stop("Check the concordance between the columns of the formula and those of the data source")
    }
  }
  # Reconstituter le dataframe à partir du formula
  df = model.frame(formula = as.formula(formule), data = data)
  # sauvegarder les variables explicatives
  instance$explicatives = colnames(df[,-1])
  # variables explicatives
  X = df[,-1]
  # centrage reduction des variables explicatives
  if(centering == TRUE){
    X = centering.reduction(X)
  }
  # Creation de la colonne de biais
  X$biais = 1
  # Variable cible
  y = df[,1]
  instance$y = y
  # Initialisation des parametres du modele
  theta = runif(ncol(X))
  X = as.matrix(X)
  y = as.vector(y)
  if((mode_compute == 'parallel') && (!is.null(ncores))){
    if(is.na(batch_size)){
      # Mode Batch parallel
      instance$res <- dgsrow_batch_parallele(X,y,theta,ncores, leaning_rate, max_iter, tolerance)
    } else if(batch_size == 1){
      # Mode Online parallel
      instance$res <- dgs_minibatch_online_parallle(X,y,theta, ncores, batch_size, leaning_rate, max_iter, tolerance)
    }else{
      # Mode Mini Batch parallel
      instance$res <- dgs_minibatch_online_parallle(X,y,theta, ncores, batch_size, leaning_rate, max_iter, tolerance)
    }
  }else if(mode_compute == 'sequentiel'){
    if(is.na(batch_size)){
      # Mode Batch sequentiel
      instance$res <- dg_batch_seq(X,y,theta,leaning_rate=leaning_rate, max_iter=max_iter, tolerance=tolerance)
    } else if(batch_size == 1){
      # Mode Online sequentiel
      instance$res <- dg_batch_minibatch_online_seq(X,y,theta, batch_size=batch_size, random_state=random_state, leaning_rate=leaning_rate, max_iter=max_iter, tolerance=tolerance)
    }else{
      # Mode Mini Batch sequentiel
      instance$res <- dg_batch_minibatch_online_seq(X,y,theta, batch_size=batch_size, random_state=random_state, leaning_rate=leaning_rate, max_iter=max_iter, tolerance=tolerance)
    }
  }

  instance$probas <- sigmoid(X %*% instance$res$theta)
  instance$binary_predict <- binary_predict(instance$probas)
  instance$y_val = data.frame(TrueY=instance$y, Ypredict=instance$binary_predict, Probas=instance$probas)
  instance$err_rate=mean(instance$y_val$TrueY != instance$y_val$Ypredict)
  class(instance) <- "modele"
  return(instance)
}
