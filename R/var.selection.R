#' Features Selection
#'
#' @param X explicatives variables
#' @param y target variable
#' @param pval p-value
#' @import klaR
#'
#' @return an instance of selected features with associated statistics
#' @export
#'
#' @examples
#' \dontrun{
#'   var.selection(X,y,pval)
#' }
var.selection <- function(X,y,pval){
  res <- greedy.wilks(X, grouping = y, p.value.overall = pval)
  instance <- list()
  instance$varselect <-res$results
  class(instance) <- "features.selection"
  return(instance)
}





