

# Package load


install.packages("readr")
library(readr)
library(dplyr)




####### Normal Sample remove #######


# TCGA Normal Barcode list
normal <- c(".10A$",".10B$",".10C$",".10D$",
            ".11A$",".11B$",".11C$",".11D$",
            ".12A$",".12B$",".12C$",".12D$",
            ".13A$",".13B$",".13C$",".13D$",
            ".14A$",".14B$",".14C$",".14D$")


# Remove normal sample function
Remove_normal_sample <- function(n, data){
  
  # input data
  data
  
  # filter normal sample
  data <- data %>% filter(!grepl(n, sample_id))
  
  return(data)
}



# Train & Test dataset
tumor_sample <- TIDE_Score_TCGA19_miR_dup

for (n in normal){
  assign(paste0(c("tumor_sample")), Remove_normal_sample(n, tumor_sample))}

# save
write.csv(tumor_sample, file = "./data.csv")



# Validation dataset 
val_tumor_sample <- TIDE_Score_TCGA_Val_miR_dup

for (n in normal){
  assign(paste0(c("val_tumor_sample")), Remove_normal_sample(n, val_tumor_sample))}

# save
write.csv(val_tumor_sample, file = "./validation_data.csv")








