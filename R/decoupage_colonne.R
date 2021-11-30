#' Column Decoupage
#'
#' @param data dataframe to cut by column
#' @param ncores  Cutting in ncores parts
#'
#' @return list of data frame
#' @export
#'
#' @examples
#' \dontrun{
#'  decoupage_colonne(data, ncores)
#' }
decoupage_colonne<-function(data, ncores){
  set.seed(1)
  npart = floor(ncol(data)/ncores)
  data_list = list()
  last_iter = FALSE
  i <- 1
  for (start in seq(from=1, to=ncol(data), npart)){
    stop = start+(npart-1)
    if(stop >= ncol(data)){
      stop = ncol(data)
      last_iter = TRUE
    }

    data_list[[i]] <- data[,start:stop]
    i <- i + 1
    if(last_iter == TRUE){
      break
    }
  }
  df <- data.frame()
  if(ncol(data)%%ncores != 0){
    df = as.data.frame(data_list[[ncores+1]])
    for(i in 1:ncol(df)){
      data_list[[i]] <- cbind(data_list[[i]],df[,i])
    }
    data_list[[ncores+1]] <- NULL
  }
  return(data_list)
}
