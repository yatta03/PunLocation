data len: total / hetero / homo  
total: 4030: 1780 + 2250  
train: 2901: 1281 + 1620  
val: 726: 321 + 405  
test: 403: 178 + 225

10-fold cross validation，training 再拆 0.2 作為 validation  
=> 0.9\*0.8 / 0.9\*0.2 / 0.1 trainind / validation / test data  
utils.py: get_n_fold_split

## result

all the result is evaluated with 10-fold cross validation.

### separately trained model:

| data   | Task              | Instances | Precision | Recall | F1     | Accuracy |
| ------ | ----------------- | --------- | --------- | ------ | ------ | -------- |
| Hetero | CV Classification | 1780      | 88.164    | 91.424 | 89.764 | 85.112   |
| Hetero | CV Location       | 1271      | 77.711    | 71.046 | 74.229 | 71.046   |
| Homo   | CV Classification | 2250      | 89.846    | 90.853 | 90.347 | 86.133   |
| Homo   | CV Location       | 1607      | 82.671    | 75.109 | 78.709 | 75.109   |

### Our Model (trained with homo and hetero together):

| data   | Task              | Instances | Precision | Recall | F1     | Accuracy |
| ------ | ----------------- | --------- | --------- | ------ | ------ | -------- |
| all    | CV Classification | 4030      | 91.414    | 92.113 | 91.762 | 88.189   |
| all    | CV Location       | 2878      | 84.119    | 77.484 | 80.666 | 77.484   |
| Hetero | CV Classification | 1780      | 92.504    | 91.267 | 91.881 | 88.483   |
| Hetero | CV Location       | 1271      | 88.190    | 80.488 | 84.163 | 80.488   |
| Homo   | CV Classification | 2250      | 93.555    | 91.226 | 92.376 | 89.244   |
| Homo   | CV Location       | 1607      | 86.426    | 78.843 | 82.460 | 78.843   |

It seems that model trained with both type of data together can achieve an overall better performance.
