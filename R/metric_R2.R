#' coefficient of determination R2
#'
#' This function measures the performance of the model in terms of recognition rate.
#'
#' @param y the true y
#' @param ypred the predict y
#'
#' @return accuracy or recognition rate
#' @export
#'
#' @author "Saliou NDAO <salioundao21@gmail.com>"
#' @examples
#' \dontrun{
#'    metric_R2(y, ypred)
#' }
#'
metric_R2 <- function(y, ypred){
  u = sum((y-ypred)**2)
  v = sum((y-mean(y))**2)
  return(1-u/v)
}


