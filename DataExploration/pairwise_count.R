library(igraph)
library(ggplot2)
library(dplyr)
library(reshape2)


get_edges<-function(x){
  as.data.frame(t(combn(x,2)))
}


pairwise_count<-function(x){
  
#For each group, get all pairwise combination of members
  m<-lapply(x,function(y){
    as.data.frame(t(combn(y,2)))
  })
  
#Bind groups into a dataframe and give it a count column
  df<-bind_rows(m)
  colnames(df)<-c("Point1","Point2")
  return(df)
}

consensus_set<-function(graphs){
  alldf<-lapply(graphs,pairwise_count)
  alldf<-melt(alldf,id.vars=c("Point1","Point2"))
  
}


#Get consensus probability that a pair are in the same set.
library(dplyr)
alldf %>% group_by(Point1,Point2) %>% summarize(n=n()/3)

# How to choose final sets?
