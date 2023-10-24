
# Package load 


install.packages("readr")
library(readr)
library(dplyr)



# annotation gene symbol

BiocManager::install("org.Hs.eg.db")

library(org.Hs.eg.db)

gene_symbol <- mapIds(org.Hs.eg.db, keys = TCGA_gene_id_rm$Ensembl_ID, 
                      keytype = "ENSEMBL", column = "SYMBOL")


gene_symbol_df <- data.frame(gene_symbol, TCGA_gene_id)

gene_symbol_df_na <- na.omit(gene_symbol_df)



# duplicated genes list 

gene_symbol_dup1 <- sort(gene_symbol_df_na$gene_symbol)

gene_symbol_dup2 <- duplicated(gene_symbol_dup1)

gene_symbol_dup3 <- unique(gene_symbol_dup1[gene_symbol_dup2])

gene_symbol_dup <- as.data.frame(gene_symbol_dup3)
colnames(gene_symbol_dup) <- c("gene_symbol")



# non-duplicating genes list **

gene_symbol_uniq1 <- sort(gene_symbol_df_na$gene_symbol)

gene_symbol_uniq2 <- unique(gene_symbol_uniq1)

gene_symbol_uniq3 <- as.data.frame(gene_symbol_uniq2)
colnames(gene_symbol_uniq3) <- c("gene_symbol")
  
# function
non_dup_gene <- function(n, data){
  # input data
  data
  
  # filter duplicating genes
  data <- data %>% filter(!gene_symbol %in% n)
  
  return(data)
}

gene_symbol_uniq <- gene_symbol_uniq3 #gene_symbol_df_na

for (n in gene_symbol_dup3){
  assign(paste0(c("gene_symbol_uniq")), non_dup_gene(n, gene_symbol_uniq))}






# TCGA gene expression data processing function

gene_pheno_processing <- function(i){
  fpkm_uq <- paste0(c("TCGA-"), i, c(".htseq_fpkm-uq.tsv"))
  phenotype <- paste0(c("TCGA-"), i, c(".GDC_phenotype.tsv"))
  
  # processing TCGA gene expression data
  gene_exp <- readr::read_tsv(fpkm_uq)
  
  # Pheno data [immunotherapy data]
  pheno <- readr::read_tsv(phenotype)
  immuno <- data.frame(pheno$submitter_id.samples, pheno$prior_treatment.diagnoses)
  colnames(immuno)[c(1,2)] <- c("sample_id", "prior_therapy")
  prior_immuno <- immuno %>% filter(prior_therapy == "No")
  
  #merge gene expression data and pheno data
  immunotherapy <- which(colnames(gene_exp[-1]) %in% prior_immuno$sample_id)
  TCGA_gene_immuno <- gene_exp[,immunotherapy]
  TCGA_gene_immuno <- cbind(TCGA_gene_immuno, gene_symbol_df)
  TCGA_gene_immuno <- na.omit(TCGA_gene_immuno)
  
  # gene expression data Normalization
  n <- ncol(TCGA_gene_immuno) - 2
  
  TCGA_gene_immuno_nor <- TCGA_gene_immuno[,c(2:n)]
  
  mean <- colMeans(TCGA_gene_immuno_nor)
  mean2 <- as.numeric(mean)
  
  TCGA_gene_immuno_nor2 <- sweep(TCGA_gene_immuno_nor,2,STATS=mean2,
                                 FUN = "-")
  
  TCGA_gene_immuno_nor2 <- cbind(TCGA_gene_immuno_nor2, 
                                 TCGA_gene_immuno$gene_symbol)
  
  colnames(TCGA_gene_immuno_nor2)[c(n)] <- c("gene_symbol")
  
  
  # Duplicate genes Mean Calculation and Input
  combine <- data.frame()
  
  gene_dup_list = as.character(gene_symbol_dup$gene_symbol)
  
  for (n in gene_dup_list){
    dup <- TCGA_gene_immuno_nor2 %>% filter(gene_symbol == n)
    dup <- subset(dup, select=-gene_symbol)
    MEAN <- colMeans(dup)
    MEAN2 <- as.numeric(MEAN)
    
    dup <- rbind(dup, MEAN2)  
    dup <- dup[nrow(dup),]
    
    dup_gene <- data.frame(gene_symbol=c(n))
    MEAN3 <- cbind(dup_gene, dup)
    
    combine <- rbind(combine, MEAN3)
  }
  
  
  # Duplicate genes remove
  TCGA_gene_immuno_nor3 <- merge(TCGA_gene_immuno_nor2, gene_symbol_uniq,
                                by="gene_symbol")
  
  
  # Duplicate Gene values merge
  TCGA_gene_immuno_nor4 <- rbind(TCGA_gene_immuno_nor3, combine)
  
  return(TCGA_gene_immuno_nor4)
}




#######################################################################
## 1. Melanoma (TCGA gene expression data processing each tumor type) #

for (i in c("SKCM","UVM")){
  assign(paste0(i, c("_gene_nor_dup")), gene_pheno_processing(i))}



#######################################################################
### 2. NSCLC (TCGA gene expression data processing each tumor type) ###

for (i in c("LUAD","LUSC")){
  assign(paste0(i, c("_gene_nor_dup")), gene_pheno_processing(i))}



#######################################################################
### 3. Other (TCGA gene expression data processing each tumor type) ###

for (i in c("BLCA","BRCA","CESC","COAD","ESCA",
            "HNSC","KIRC","KIRP","LGG","LIHC",
            "OV","PAAD","SARC","STAD", "UCEC")){
  assign(paste0(i, c("_gene_nor_dup")), gene_pheno_processing(i))}



###################################################################################
### 4. Other(Validation) (TCGA gene expression data processing each tumor type) ###


for (i in c("ACC","CHOL","DLBC","KICH","MESO",
            "PCPG","PRAD","READ","TGCT","THCA",
            "THYM","UCS")){
  assign(paste0(i, c("_gene_nor_dup")), gene_pheno_processing(i))}



