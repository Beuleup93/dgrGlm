#' Sigmoid Function
#'
#' @param z linear combination of predictor variables
#'
#' @export
#' @return a vector containing the probabilities of assignment to the positive or negative class
#'
#' @examples
#' \dontrun{
#'   sigmoid(z)
#' }
sigmoid <- function(z){
  return (1/(1+exp(-z)))
}
