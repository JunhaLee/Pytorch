# Pytorch
## Practice of implementing some CNN algorithms using Pytorch

### 1. Alexnet
### 2. VGG
### 3. ResNET
------------
### Data Configuration

> **Class = (hat, outer, top, bottom, shoes)**
> 
> **Train set**
>> hat : 210
>> 
>> outer : 4984
>> 
>> top : 20218
>> 
>> bottom : 7304
>> 
>> shoes : 454

------------
### Model Version

> **version 1** : First version for classification
> 
> **version 2** : Editing loss function and loss ratio. && Adding code for Confusion Matrix    
> * I can check about data im-balance problem. Because of lack for class 1 (outer), confusion is caused between class 1 (outer) and class 2 (top)
> 
> **version 3** : Adding augmentation dataset about class 1 (outer). Using validation set to training    
> * ResNET not work well (maybe) because of overfitting. In case of other models, there are no great effect to total accuracy
> 
> **version 4** : Adding augmentation dataset for all training dataset   
