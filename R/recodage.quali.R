#' Recoding target variable
#'
#' @param y qualitative variable to coded
#'
#' @author Saliou NDAO
#'
#' @return a dataframe where each modality represents a binary column
#' @export
#'
#' @examples
#' recodage.quali(iris$Species)
recodage.quali<- function(y){
  # LIST OF Y
  y <- as.factor(y)
  list_y <- list()
  j<-1
  for (i in unique(y)){
    list_y[[j]]<- ifelse(y==i,1,0)
    j<-j+1
  }
  df_modalite<- as.data.frame(sapply(list_y,function(x) x))
  colnames(df_modalite) <- unique(y)
  return (df_modalite)
}

#' Re-coding Features
#'
#' @param X data frame
#'
#' @import PCAmixdata
#' @import tidytable
#'
#' @return A encoding data frame
#'
#' @author Saliou NDAO
#' @export
#'
#' @examples
#' recodage_X(iris)
recodage_X <- function(X){
  decoupage_X <- splitmix(X)
  encode_X <- get_dummies.(decoupage_X$X.quali, drop_first = FALSE)
  encode_X <- splitmix(encode_X)
  return(cbind(decoupage_X$X.quanti,encode_X$X.quanti))
}







