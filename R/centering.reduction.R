#' centering and reduction
#'
#' @param X the dataframe to center and reduce
#'
#' @importFrom stats var
#' @return the dataframe centered and reduce
#' @export
#'
#' @examples
#' centering.reduction(iris[,-ncol(iris)])
centering.reduction <- function(X){
  one.cr <- function(x){
    n<-length(x)
    moy<-mean(x)
    ect <- sqrt((n-1)/n*var(x))
    y<- (x-moy)/ect
    return(y)
  }
  Y <- as.data.frame(lapply(X, one.cr))
  return(Y)
}

