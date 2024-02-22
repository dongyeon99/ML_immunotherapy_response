
# Package load 

library(readr)
library(dplyr)


# processing function
PCAWG_TIDE_miRNA <- function(i){
  # load data
  score <- paste0(c("PCAWG_TIDE_result.csv"))
  miRNA <- paste0(c("x3t2m1.mature.mirna.all.matrix.log"))
  
  # TIDE Score data
  tide_score <- read.csv(score)
  
  TIDE_Score <- data.frame(tide_score$Patient,
                           tide_score$Responder,
                           tide_score$TIDE,
                           tide_score$Dysfunction,
                           tide_score$Exclusion,
                           tide_score$CTL.flag)
  
  colnames(TIDE_Score)[c(1:6)] <- c("sample_id", "Responder",
                                    "TIDE","Dysfunction","Exclusion","CTL.flag")
  
  # miRNA expression data
  pcawg_miRNA <- read.csv(miRNA, sep = '\t', header = TRUE)
  PCAWG_miRNA <- t(pcawg_miRNA)
  miRNA_ID <- PCAWG_miRNA[c(1),]
  colnames(PCAWG_miRNA) = miRNA_ID
  PCAWG_miRNA <- PCAWG_miRNA[-1,]
  sample_id <- rownames(PCAWG_miRNA)
  PCAWG_miRNA <- cbind(PCAWG_miRNA, sample_id)
  
  # TCGA miRNA + TIDE Score
  TIDE_Score_PCAWG_miR <- merge(PCAWG_miRNA, TIDE_Score, by="sample_id")
  
  return(TIDE_Score_PCAWG_miR)
}


# run function
PCAWG_TIDE_miR <- PCAWG_TIDE_miRNA(i)


# data save 
write.csv(PCAWG_TIDE_miR, file = "PCAWG_TIDE_miR.csv")
