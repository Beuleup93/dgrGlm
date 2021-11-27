#' Binary or probabilities prediction
#'
#' @param model the model created with fit function logistic regression
#' @param new_data the new data to be classified
#' @param type_pred the type of prediction to perform (binary prediction or probability prediction) by default 'CLASS'
#' @param centering to center and reduce the variables, by default FALSE
#'
#' @return an instance containing the binary or probability prediction.
#' @export
#'
#' @examples
#' \dontrun{
#'  predict(model, new_data, type_pred, centering)
#' }
dgrglm.predict <- function(model, new_data, type_pred='CLASS', centering=FALSE){
  instance = list()
  # verifier que toutes les variables du modéle y figure
  cols_modele = model$explicatives
  for (current_col in colnames(new_data)){
    if(!is.element(current_col, cols_modele)){
      print(paste(current_col,"is not part of the variables used to build the model"))
      stop("Check the consistency of the new variables with that of the model")
    }
  }
  # enleve la colonne cible
  if(centering == TRUE){
   new_data = centering.reduction(new_data)
  }
  # Ajoutons la colonne de biais
  new_data$biais = 1
  theta = model$res$theta
  PI <- sigmoid(as.matrix(new_data) %*% as.vector(theta))
  if(type_pred == 'CLASS'){
    instance$binary_predict <- binary_predict(PI)
  }else{
   instance$probas <- PI
  }
  class(instance) <- "predict"
  return(instance)
}

