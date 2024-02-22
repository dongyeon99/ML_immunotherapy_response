
# Package load 


install.packages("readr")
library(readr)
library(dplyr)



# TIDE Results + TCGA miRNA expression data function 

TIDE_miRNA <- function(i){
  # load data
  score <- paste0(c("TIDE_Score_"), i, c("_dup.csv"))
  miRNA <- paste0(c("TCGA-"), i, c(".mirna.tsv"))
  
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
  tcga_miRNA <- read.csv(miRNA, sep = '\t', header = TRUE)
  TCGA_miRNA <- t(tcga_miRNA)
  miRNA_ID <- TCGA_miRNA[c(1),]
  colnames(TCGA_miRNA) = miRNA_ID
  TCGA_miRNA <- TCGA_miRNA[-1,]
  sample_id <- rownames(TCGA_miRNA)
  TCGA_miRNA <- cbind(TCGA_miRNA, sample_id)
  
  # TCGA miRNA + TIDE Score
  TIDE_Score_TCGA_miR <- merge(TCGA_miRNA, TIDE_Score, by="sample_id")
  
  return(TIDE_Score_TCGA_miR)
}





###########################################################3
################## 1. Melanoma ##########################3##
# TCGA miR + TIDE_Score data processing each tumor type

for (i in c("SKCM","UVM")){
  assign(paste0(c("TIDE_Score_TCGA_"), i, c("_miR_dup")), TIDE_miRNA(i))}


###########################################################3
################## 2. NSCLC ##########################3##
# TCGA miR + TIDE_Score data processing each tumor type

for (i in c("LUAD","LUSC")){
  assign(paste0(c("TIDE_Score_TCGA_"), i, c("_miR_dup")), TIDE_miRNA(i))}


###########################################################3
################## 3. Other ##########################3##
# TCGA miR + TIDE_Score data processing each tumor type

for (i in c("BLCA","BRCA","CESC","COAD","ESCA",
            "HNSC","KIRC","KIRP","LGG","LIHC",
            "OV","PAAD","SARC","STAD", "UCEC")){
  assign(paste0(c("TIDE_Score_TCGA_"), i, c("_miR_dup")), TIDE_miRNA(i))}



# Total 19 TCGA tumor type miRNA + TIDE Score data merge

TIDE_Score_TCGA19_miR_dup <- rbind(TIDE_Score_TCGA_BLCA_miR_dup, TIDE_Score_TCGA_BRCA_miR_dup,
                               TIDE_Score_TCGA_CESC_miR_dup, TIDE_Score_TCGA_COAD_miR_dup,
                               TIDE_Score_TCGA_ESCA_miR_dup, TIDE_Score_TCGA_HNSC_miR_dup,
                               TIDE_Score_TCGA_KIRC_miR_dup, TIDE_Score_TCGA_KIRP_miR_dup,
                               TIDE_Score_TCGA_LGG_miR_dup, TIDE_Score_TCGA_LIHC_miR_dup,
                               TIDE_Score_TCGA_LUAD_miR_dup, TIDE_Score_TCGA_LUSC_miR_dup,
                               TIDE_Score_TCGA_OV_miR_dup, TIDE_Score_TCGA_PAAD_miR_dup,
                               TIDE_Score_TCGA_SARC_miR_dup, TIDE_Score_TCGA_SKCM_miR_dup,
                              TIDE_Score_TCGA_STAD_miR_dup, TIDE_Score_TCGA_UCEC_miR_dup,
                               TIDE_Score_TCGA_UVM_miR_dup)




#####################################################################3
################## 4. Other (Validation) ##########################3##
# TCGA miR + TIDE_Score data processing each tumor type

for (i in c("ACC","CHOL","DLBC","KICH","MESO",
            "PCPG","PRAD","READ","TGCT","THCA",
            "THYM","UCS")){
  assign(paste0(c("TIDE_Score_TCGA_"), i, c("_miR_dup")), TIDE_Score_miRNA(i))}



# Validation 12 TCGA tumor type miRNA + TIDE Score data merge

TIDE_Score_TCGA_Val_miR_dup <- rbind(TIDE_Score_TCGA_ACC_miR_dup, TIDE_Score_TCGA_CHOL_miR_dup,
                                   TIDE_Score_TCGA_DLBC_miR_dup, TIDE_Score_TCGA_KICH_miR_dup,
                                   TIDE_Score_TCGA_MESO_miR_dup, TIDE_Score_TCGA_PCPG_miR_dup,
                                   TIDE_Score_TCGA_PRAD_miR_dup, TIDE_Score_TCGA_READ_miR_dup,
                                   TIDE_Score_TCGA_TGCT_miR_dup, TIDE_Score_TCGA_THCA_miR_dup,
                                   TIDE_Score_TCGA_THYM_miR_dup, TIDE_Score_TCGA_UCS_miR_dup)






