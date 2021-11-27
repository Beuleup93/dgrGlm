#' Binary or probabilities prediction
#'
#' @param model the model created with fit function logistic regression
#' @param new_data the new data to be classified
#' @param type_pred the type of prediction to perform (binary prediction or probability prediction) by default 'CLASS'
#'
#' @return an instance containing the binary or probability prediction.
#' @export
#'
#' @examples
#' \dontrun{
#'  predict(model, new_data, type_pred)
#' }
dgrglm.multiclass.predict <- function(model, new_data, type_pred='CLASS'){
  instance = list()

  # CHECK THAT ALL THE VARIABLES OF THE MODEL ARE INCLUDED
  cols_modele = model$explicatives
  for (current_col in colnames(new_data)){
    if(!is.element(current_col, cols_modele)){
      print(paste(current_col,"is not part of the variables used to build the model"))
      stop("Check the consistency of the new variables with that of the model")
    }
  }

  # ADD THE BIAIS COLUMN
  new_data$biais = 1
  df_theta = model$df_theta
  list_pred_probas <- list()
  j <- 1
  for (i in colnames(df_theta)){
    theta <- df_theta[,i]
    PI <- sigmoid(as.matrix(new_data) %*% as.vector(theta))
    list_pred_probas[[j]] <- PI
    j <- j+1
  }

  df_pred_probas<- as.data.frame(sapply(list_pred_probas,function(x) x))
  colnames(df_pred_probas) <- colnames(df_theta)

  if(type_pred == 'CLASS'){
    instance$binary_predict <- sapply(list_pred_probas,binary_predict)
  }else if(type_pred == 'PROBAS'){
    instance$probas <- df_pred_probas
  }else{
    stop("Prediction Type non disponible")
  }
  class(instance) <- "predict"
  return(instance)
}
