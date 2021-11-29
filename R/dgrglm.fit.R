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
#' @param feature_selection parameters indicating the choice to make the selection of variable or not
#' @param p_value selection criteria
#' @param rho hyper parameter which allows arbitration between RDIGE and LASSO. Elasticnet case.
#' @param C parameter allowing to arbitrate between the penalty and the likelihood in the guidance of the modeling.Elasticnet case.
#' @param iselasticnet for Elasticnet
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
#'  dgrglm.fit(formule, data,ncores=3, mode_compute="parallel")
#' }
dgrglm.fit <- function(formule, data, ncores=NA, mode_compute="parallel", leaning_rate=0.1,
                       max_iter=100, tolerance=1e-04, batch_size=NA,
                       random_state=102, centering = FALSE, feature_selection=FALSE,
                       p_value=0.01, rho=0.1, C=0.1, iselasticnet=FALSE){

  # OBJECT S3
  instance <- list()

  # CONTROL OF USER INPUTS
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

  if (C < 0){
    stop("'C' must be positive")
  }

  if (rho < 0){
    stop("'Rho' must be positve")
  }

  # CONTROL BATCH SIZE FOR EACH COMPUTE MODE
  if((mode_compute=="parallel") && (!is.na(batch_size))){
    if((dim(data)[1] %% ncores == 0) && ((batch_size <=  0) || (batch_size > (dim(data)[1]%/% ncores)))){
      stop("'Batch size' must be between 1 and nbObs for each core ")
    }
  } else if((mode_compute=="sequentiel") && (!is.na(batch_size))){
    if(((batch_size <=  0) || (batch_size > (dim(data)[1])))){
      stop("'Batch size' must be between 1 and length of data ")
    }
  }

  if(is.na(ncores) || ncores<=0 || ncores>=detectCores()){
    ncores = detectCores()-1
  }else{
    stop('enter a correct value for number of ncore')
  }

  # COLUMN MATCHING CONTROL BETWEEN DATA AND FORMULA
  f = formula(formule)
  colonne_names = colnames(data)
  for (v in all.vars(f)){
    if(!is.element(v, colonne_names) && v != '.'){
      print(paste("Whoops!! Correspondence error between variables:: -->", v))
      stop("Check the concordance between the columns of the formula and those of the data source")
    }
  }

  # RECONSITUTE DATAFRAME FROM THE FORMULA
  df = model.frame(formula = as.formula(formule), data = data)

  # SHUFLE DATA
  if(!is.null(random_state)){
    set.seed(seed = random_state)
    rows <- sample(nrow(df))
    df <- df[rows, ]
  }
  # REMOVE NA in DATASET
  df <- na.omit(df)

  # TARGET AND EXPLICATIVES VARIABLES
  y = df[,1]
  X = df[,-1]

  # CENTERING AND REDUCTION EXPLICATIVES VARIABLE
  if(centering == TRUE){
    scaling = centering.reduction(X)
    X = scaling$Y
    instance$archive_EctMoy <- scaling$Arch
  }else{
    instance$archive_EctMoy <- NULL
  }

  if(feature_selection==TRUE && !is.null(p_value)){
    sele <- var.selection(X,y,p_value)
    X <- X[,sele$varselect$vars]
  }

  # SAVE EXPLICATIVES VARIABLE IN INSTANCE
  instance$explicatives = colnames(X)

  # CREATE BIAIS COLUMN
  X$biais = 1

  # CONVERT DATA
  X = as.matrix(X)
  y = as.vector(y)

  # INIT COEFS MODEL
  theta = runif(ncol(X))

  if((mode_compute == 'parallel') && (!is.null(ncores))){
    if(is.na(batch_size)){
      # MODE BATCH PARALLEL
      if(iselasticnet==TRUE){
        instance$res <- dgsrow_batch_parallele(X,y,theta,ncores, leaning_rate, max_iter,tolerance, rho, C)
      }else{
        instance$res <- dgsrow_batch_parallele(X,y,theta,ncores, leaning_rate,max_iter, tolerance)
      }

    } else if(batch_size == 1){
      # MODE ONLINE PARALLEL
      if(iselasticnet==TRUE){
        instance$res <- dgs_minibatch_online_parallle(X,y,theta, ncores, batch_size,leaning_rate, max_iter, tolerance, rho, C)
      }else{
        instance$res <- dgs_minibatch_online_parallle(X,y,theta, ncores, batch_size,leaning_rate, max_iter, tolerance)
      }

    }else{
      # MODE MINI BATCH PARALLEL
      if(iselasticnet==TRUE){
        instance$res <- dgs_minibatch_online_parallle(X,y,theta, ncores, batch_size, leaning_rate, max_iter, tolerance, rho, C)
      }else{
        instance$res <- dgs_minibatch_online_parallle(X,y,theta, ncores, batch_size, leaning_rate, max_iter, tolerance)
      }

    }

  }else if(mode_compute == 'sequentiel'){
    if(is.na(batch_size)){
      # MODE BATCH SEQUENTIEL
      if(iselasticnet==TRUE){
        instance$res <- dg_batch_seq(X,y,theta,leaning_rate=leaning_rate, max_iter=max_iter, tolerance=tolerance,rho, C)
      }else{
        instance$res <- dg_batch_seq(X,y,theta,leaning_rate=leaning_rate, max_iter=max_iter, tolerance=tolerance)
      }

    } else if(batch_size == 1){
      # MODE ONLINE SEQUENTIEL
      if(iselasticnet==TRUE){
        instance$res <- dg_batch_minibatch_online_seq(X,y,theta, batch_size=batch_size,random_state=random_state,
                                                      leaning_rate=leaning_rate,max_iter=max_iter, tolerance=tolerance, rho, C)
      }else{
        instance$res <- dg_batch_minibatch_online_seq(X,y,theta, batch_size=batch_size, random_state=random_state,
                                                      leaning_rate=leaning_rate, max_iter=max_iter, tolerance=tolerance)
      }
    }else{
      # MODE MINI BATCH SEQUENTIEL
      if(iselasticnet==TRUE){
        instance$res <- dg_batch_minibatch_online_seq(X,y,theta, batch_size=batch_size,random_state=random_state,
                                                      leaning_rate=leaning_rate, max_iter=max_iter, tolerance=tolerance, rho, C)
      }else{
        instance$res <- dg_batch_minibatch_online_seq(X,y,theta, batch_size=batch_size, random_state=random_state,
                                                      leaning_rate=leaning_rate, max_iter=max_iter, tolerance=tolerance)
      }

    }
  }else{
    stop("You must enter execution mode: 'sequentiel or parallel")
  }

  instance$probas <- sigmoid(X %*% instance$res$theta)
  instance$binary_predict <- binary_predict(instance$probas)
  instance$y_val = data.frame(TrueY=y, Ypredict=instance$binary_predict, Probas=instance$probas)
  instance$err_rate=mean(instance$y_val$TrueY != instance$y_val$Ypredict)
  instance$metric <- metrics(instance$probas,y,ncol(X))
  class(instance) <- "modele"
  return(instance)
}

