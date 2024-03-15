# Identifying microRNAs associated with tumor immunotherapy response using an interpretable machine learning model
Tumor immunotherapy is an anti-cancer treatment strategy that has the benefits of continuous clinical response and fewer side effects. 
However, the high cost and the fact that some patients do not respond to treatment present challenges treatment. 
In this study, we built machine learning models aimed at predicting responses to tumor immunotherapy by employing miRNA expression profiles.
To interpret the model's underlying mechanisms, we assessed the contribution of each miRNA to the prediction process using Shapley Additive explanations (SHAP).
In conclusion, this study successfully predicted responses to tumor immunotherapy using previously unexplored types of miRNAs profiles data. 


<br/>

## Data processing
Use the R script files in Data/Preprocessing/ folder to create data for TIDE tools and data for machine learning.

The miRNA expression data for the TCGA cohorts can be accessed at https://gdc.xenahubs.net. 

Additionally, the gene expression data for TCGA cohorts, which were used as input data for the TIDE method, also can be found at https://gdc.xenahubs.net. 

For additional validation, bulk RNA-seq and mature miRNA expression data were acquired from PCAWG (Pan-Cancer Analysis of Whole Genomes) (https://pcawg.xenahubs.net).


<br/>

## Download
The machine learning model and data used for training have been uploaded to this repository.

Because the file size is limited to 25MB or less, the data used for the training was uploaded to the gz file. So use it after decompressing.


<br/>

## Installation
### Dependencies

```
python
pandas
numpy
matplotlib
warnings
os
sklearn
joblib
auto_shap
```


<br/>

## Citation

Dong-Yeon Nam and Je-Keun Rhee. "Identifying microRNAs associated with tumor immunotherapy response using an interpretable machine learning model." Sci Rep 14, 6172 (2024). https://doi.org/10.1038/s41598-024-56843-3


<br/>
