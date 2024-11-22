data len: total / hetero / homo  
total: 4030: 1780 + 2250  
train: 2901: 1281 + 1620  
val: 726: 321 + 405  
test: 403: 178 + 225

10-fold cross validation，training 再拆 0.2 作為 validation  
=> 0.9\*0.8 / 0.9\*0.2 / 0.1 trainind / validation / test data  
utils.py: get_n_fold_split
