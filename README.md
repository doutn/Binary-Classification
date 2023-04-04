### Binary-Classification
--- 
강아지와 고양이를 분류하는 프로젝트입니다. 모델은 **VGGNet**을 사용했으며, **Pytorch**를 기반으로 코드가 작성되었습니다.

### Dataset
---
Dataset은 Kaggle의 [Cats and Dogs](https://www.kaggle.com/datasets/tongpython/cat-and-dog)데이터셋을 사용했습니다.  

train dataset은 약 8000장이고, test dataset은 약 2000장입니다. 별다른 valid dataset은 사용하지 않았습니다.

### Train
---
**RTX3060**을 기준으로는 학습시간이 **약 5분** 걸렸으며, **MPU**를 기준으로는 **약 12분** 걸렸습니다.  
모델의 총 **파라미터** 수는 **39,905,346**개이며, lowest_loss는 **0.0006** 그리고 best_acc는 **100.00%** 입니다.

### Inference 
---
**utils.py** 파일 안에 있는 predict 함수를 통해서 학습된 모델을 추론해볼 수 있습니다.  
**Image**와 **model_pth**를 입력받아서, 고양이 혹은 강아지를 **print** 합니다.