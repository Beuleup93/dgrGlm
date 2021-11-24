#' Algorithm to cut data
#'
#' @param data data to cut
#' @param ncores split data ncores parts
#' @return a dataframe with an additional column indicating the group of each observation
#' @export
#'
#' @examples
#' \dontrun{
#'  decoupage_ligne(data,ncores)
#' }
decoupage_ligne <- function(data, ncores){
  set.seed(1)
  random <- sample(1:nrow(data), replace = FALSE, nrow(data))
  data$fold <- ceiling(random/(nrow(data)/ncores))
  t <- table(data$fold)
  if(max(t)>min(t)){
    df = as.data.frame(t)
    ind = df[df$Freq == unique(max(df$Freq)),]$Var1
    if(length(ind)>1){
      for (i in 1:length(ind)){
        data[data$fold==as.integer(ind[i]),][i,i]<- NA
      }
    }else{
      data[data$fold==as.integer(ind),][1,]<- NA
    }
    data = na.omit(data)
  }
  return (data)
}
