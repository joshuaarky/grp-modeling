# grp-modeling
This repository contains all code as well as the tabular dataset used in the project "Modeling per capita gross regional product using remotely sensed data."

For replication, start with the folder rs_preprocessing, which contains all scripts for the preprocessing of RS data. The Zhang et al. (2024) and DOSE V2.11 features do not require preprocessing, since they are already provided at the sub-national level. Run all scripts individually, saving 'rs_feature_preprocessing.py' for last.

Once the tabular dataset has been created with 'rs_feature_preprocessing.py', the scripts in the modeling folder can be run. Each script corresponds to one model family. Each model family contains seven feature combinations (labeled 'a' through 'g'). Each script trains each model in the famil and runs SHAP feature analysis as well as creates SHAP feature importance plots.

Other plots that appear in the manuscript can be created using the scripts in the plotting folder.
