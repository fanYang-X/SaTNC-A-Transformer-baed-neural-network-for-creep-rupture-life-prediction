# SaTNC: A Transformer-baed neural network for creep rupture life prediction  

By [Fan Yang](https://github.com/fanYang-X), [Wenyue Zhao](https://shi.buaa.edu.cn/09652/zh_CN/index.htm).

## Background  
With the development of material informatics, it becomes more important to introduce physical information constraints into the ML model. Here, a neural network with physical information constraints (Transformer-based) is designed for the prediction of creep rupture life, which is concerned in alloy design.

## Updata

***05/2/2023***
Initial commits:

1. Creep data, including creep datasets (.csv).  
   Note, except alloying elements features (wt.%), the features with "_L12" and "_A1" postfix are contents (at.) of alloying elements in γ'/γ, which calculated by ThermoCalc.
2. SaTNC model code
3. ML model, including SVR, RF, LightGBM, DCSA (refer to https://github.com/wujunming1/mla-shu)

## Usage 

The versions of the pyhton library used are as follows:  
pandas -- 1.3.1  
numpy -- 1.20.3  
scikit-learn -- 1.1.2  
lightgbm -- 3.2.1  
torch -- 1.9.0
