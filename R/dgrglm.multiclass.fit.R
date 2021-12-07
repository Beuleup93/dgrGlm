#' Logistic regression multiclass
#'
#' This function allows us to create a binary logistic regression model
#'
#' @param formule allows you to define the target variable and predictor variables
#' @param data the data source containing all the variables specified in the formula
#' @param leaning_rate is the learning rate that controls the magnitude of the vector update
#' @param max_iter is the number of iterations
#' @param tolerance an additional parameter which specifies the minimum movement allowed for each iteration
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
#'  dgrglm.multiclass.fit(formule, data)
#' }
dgrglm.multiclass.fit <- function(formule, data, leaning_rate=0.1, max_iter=3000, tolerance=1e-04, random_state=102, centering = FALSE){

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

  # SAVE EXPLICATIVES VARIABLE IN INSTANCE
  instance$explicatives = colnames(df[,-1])

  # TARGET AND EXPLICATIVES VARIABLES
  y = df[,1]
  X = df[,-1]

  # CENTERING AND REDUCTION EXPLICATIVES VARIABLE
  if(centering == TRUE){
    X = centering.reduction(X)
  }

  # CREATE BIAIS COLUMN
  X$biais = 1

  # DATAFRAME OF BINARY VARIABLE MODALITY
  df_modalite<- recodage.quali(y)

  # SHUFLE DATAFRAME MODALITY
  if(!is.null(random_state)){
    set.seed(seed = random_state)
    rows <- sample(nrow(df_modalite))
    df_modalite <- df_modalite[rows, ]
  }

  # DATAFRAME THETA
  list_theta = list()
  for(i in 1:length(unique(y))){
    list_theta[[i]] <- runif(ncol(X))
  }
  df_theta<- as.data.frame(sapply(list_theta,function(x) x))
  colnames(df_theta) <- unique(y)

  # CONVERT DATA
  X = as.matrix(X)

  # LOGISTIC REGRESSION IMPLEMENTATION FOR EACH CLASS
  iter <- 1
  while(iter <= max_iter) {
    for(i in 1:length(unique(y))){
      theta_mod <- df_theta[,i]
      y_mod <- df_modalite[,i]
      grad = gradient(theta_mod, as.matrix(X), y_mod)
      new_theta = theta_mod - leaning_rate*grad
      df_theta[,i] <- new_theta
    }
    iter<- iter+1
  }

  # AT THIS LEVEL, WE HAVE THE ESTIMATED COEFFICIENTS IN DF_THETA
  list_pred_probas <- list()
  j <- 1
  for (i in colnames(df_theta)) {
    theta <- df_theta[,i]
    PI <- sigmoid(as.matrix(X) %*% as.vector(theta))
    list_pred_probas[[j]] <- PI
    j <- j+1
  }
  df_pred_probas<- as.data.frame(sapply(list_pred_probas,function(x) x))
  colnames(df_pred_probas) <- colnames(df_theta)

  accuracy <- 0
  for(col in 1:ncol(df_theta)){
    for (row in 1:nrow(df_modalite)){
      if(df_modalite[row,col]==1 && df_pred_probas[row,col]>=0.5){
        accuracy <- accuracy+1
      }
    }
  }
  instance$df_theta <- df_theta
  class(instance) <- "ModeleMultiClass"
  return(instance)
}



