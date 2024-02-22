
# Package load 
install.packages(c("readr","dplyr"))

library(tidyr)
library(readr)
library(dplyr)
library(org.Hs.eg.db)


# PCAWG dataset gene symbol
pcawg <- read.table("tophat_star_fpkm_uq.v2_aliquot_gl.sp.log", sep = "\t")

pcawg <- t(pcawg)

pcawg_header <- pcawg[1,]

colnames(pcawg) <- pcawg_header

colnames(pcawg)[1] <- "sample_id"

pcawg <- pcawg[-1,]



# PCAWG tumor normal
pcawg_sample_type <- read.table("sp_specimen_type", sep = "\t", header = T)

colnames(pcawg_sample_type) <- c("sample_id", "sample_type")



# exclude normal sample
pcawg_tumor <- pcawg_sample_type %>% filter(sample_type != "Normal - blood derived")
pcawg_tumor <- pcawg_tumor %>% filter(sample_type != "Normal - solid tissue")
pcawg_tumor <- pcawg_tumor %>% filter(sample_type != "Normal - other")
pcawg_tumor <- pcawg_tumor %>% filter(sample_type != "Normal - tissue adjacent to primary")



# PCAWG data [gene expression data & tumor sample]
pcawg_fi <- merge(pcawg ,pcawg_tumor, by='sample_id')

pcawg_fi <- pcawg_fi[,-ncol(pcawg_fi_3)]

pcawg_fi <- t(pcawg_fi)


pcawg_fi_header <- pcawg_fi[1,]

colnames(pcawg_fi) <- pcawg_fi_header

pcawg_fi <- pcawg_fi[-1,]



# pcawg gene ensembl id
pcawg_gene_id <- dimnames(pcawg_fi)[1]

pcawg_gene_id <- data.frame(pcawg_gene_id)

colnames(pcawg_gene_id) <- c("Ensembl_id")

pcawg_gene_id <- tidyr::separate_rows(pcawg_gene_id, Ensembl_id, sep=".")

pcawg_gene_id <- pcawg_gene_id %>% filter(nchar(Ensembl_id) > 10)


pcawg_gene_symbol <- mapIds(org.Hs.eg.db, keys = pcawg_gene_id1$Ensembl_id, 
                            keytype = "ENSEMBL", column = "SYMBOL")

pcawg_gene_symbol_df <- data.frame(pcawg_gene_symbol, pcawg_gene_id)

pcawg_gene_symbol_df_na <- na.omit(pcawg_gene_symbol_df)


# replace gene symbol 
PCAWG_gene <- cbind(pcawg_fi, pcawg_gene_symbol_df)

PCAWG_gene_na <- na.omit(PCAWG_gene)

PCAWG_gene_na_index <- PCAWG_gene_na[PCAWG_gene_na$pcawg_gene_symbol]

rownames(PCAWG_gene_na) <- PCAWG_gene_na_index



# gene expression data Normalization
n <- ncol(PCAWG_gene_na) - 2

PCAWG_gene_na_nor <- PCAWG_gene_na[,c(1:n)]
PCAWG_gene_na_nor[, c(1:n)] <- sapply(PCAWG_gene_na_nor[, c(1:n)], FUN = "as.numeric")

mean <- colMeans(PCAWG_gene_na_nor)
mean2 <- as.numeric(mean)

PCAWG_gene_na_nor2 <- sweep(PCAWG_gene_na_nor,2,STATS=mean2,
                            FUN = "-")

PCAWG_gene_na_nor2 <- cbind(PCAWG_gene_na_nor2, 
                            PCAWG_gene_na$pcawg_gene_symbol)


colnames(PCAWG_gene_na_nor2)[c(n+1)] <- c("gene_symbol")



# Duplicate genes Mean Calculation and Input
combine <- data.frame()

PCAWG_gene_dup_list = as.character(PCAWG_gene_na_index[duplicated(PCAWG_gene_na_index)])

for (n in PCAWG_gene_dup_list){
  dup <- PCAWG_gene_na_nor2 %>% filter(gene_symbol == n)
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
PCAWG_gene_dup <- data.frame(PCAWG_gene_dup_list)
colnames(PCAWG_gene_dup) <- "gene_symbol"

PCAWG_gene_symbol_uniq <- PCAWG_gene_na_index[!duplicated(PCAWG_gene_na_index)]
PCAWG_gene_symbol_uniq <- data.frame(PCAWG_gene_symbol_uniq)
colnames(PCAWG_gene_symbol_uniq) <- "gene_symbol"


for (i in PCAWG_gene_dup_list){
  PCAWG_gene_symbol_uniq2 <- PCAWG_gene_symbol_uniq2 %>% filter(!gene_symbol == i)}


PCAWG_gene_na_nor3 <- merge(PCAWG_gene_na_nor2, PCAWG_gene_symbol_uniq2,
                            by="gene_symbol")



# Duplicate Gene values merge
PCAWG_gene_na_nor4 <- rbind(PCAWG_gene_na_nor3, combine)



# Data save
write.table(PCAWG_gene_na_nor4, file = "PCAWG_gene_data.txt", sep = "\t")


