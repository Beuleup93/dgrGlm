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
  instance <- list()
  archive.ect_mean<-function(x){
    n<-length(x)
    moy<-mean(x)
    ect <- sqrt((n-1)/n*var(x))
    return(c(moy,ect))
  }
  one.cr <- function(x){
    n<-length(x)
    moy<-mean(x)
    ect <- sqrt((n-1)/n*var(x))
    y<- (x-moy)/ect
    return(y)
  }
  Y <- as.data.frame(lapply(X, one.cr))
  Arch <- as.data.frame(lapply(X, archive.ect_mean))
  instance$Y <- Y
  instance$Arch <- Arch
  class(instance) <- "Scale"
  return(instance)
}

#' centering and reduction of the test set
#'
#' @param X the test set dataframe to center and reduce
#' @param archive_moy_ect list containing the mean and standard deviation of each column of the training set
#'
#' @return the dataframe of the test set centered and reduced
#' @export
#'
#' @examples
#' arch <- data.frame(Sepal.Length=c(mean(iris[1:100,1]),
#' sqrt((nrow(iris)-1)/var(iris[1:100,1]))),Sepal.Width=c(mean(iris[1:100,2]),
#' sqrt((nrow(iris)-1)/var(iris[1:100,2]))))
#' centering.red.pred(iris[101:150,c(1,2)],arch)
centering.red.pred <- function(X, archive_moy_ect){
  instance <- list()
  cols <- colnames(X)
  for(i in cols){
    moy <- archive_moy_ect[,i][1]
    ect <- archive_moy_ect[,i][2]
    X[i] <- (X[i] - moy)/ect
  }
  instance$Xtest <- X
  class(instance) <- "Xtestcr"
  return(instance)
}
