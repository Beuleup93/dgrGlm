#' Binary prediction
#'
#' This function performs binary predictions (0 or 1) based on the probabilities of assignments.
#'
#' @param PI is the vector containing the probabilities of assignment to the positive or negative class.
#'
#' @return a vector containing the predictions in binary form
#'
#' @export
#' @examples
#' \dontrun{
#'    binary_predict(PI)
#' }
#' @author "Saliou NDAO <salioundao21@gmail.com>"
binary_predict<- function(PI){
  return(round(PI, 0))
}
