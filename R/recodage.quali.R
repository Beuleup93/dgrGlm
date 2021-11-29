#' Recoding variable
#'
#' @param y qualitative variable to coded
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





