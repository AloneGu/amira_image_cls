# amira_image_cls

image tag web from data dir

this project use python3, keras and theano (not support tensorflow)

## data

download from https://pan.baidu.com/s/1mhCnI9i

select some cats and dogs to data/dog_vs_cat/

## run

```
python run.py
```

## docker run

use Dockerfile and set env, volume

## models

* copy densenet from https://raw.githubusercontent.com/titu1994/DenseNet/master/densenet.py
* keras official model zoo

## test output

simple cnn

```
Epoch 1/5
42/42 [==============================] - 94s - loss: 1.1062 - acc: 0.5829 - val_loss: 5.7632 - val_acc: 0.6371
Epoch 2/5
42/42 [==============================] - 93s - loss: 0.8723 - acc: 0.7091 - val_loss: 4.3352 - val_acc: 0.7154
Epoch 3/5
42/42 [==============================] - 92s - loss: 0.7825 - acc: 0.7430 - val_loss: 4.2651 - val_acc: 0.7311
Epoch 4/5
42/42 [==============================] - 92s - loss: 0.7132 - acc: 0.7687 - val_loss: 3.5350 - val_acc: 0.7807
Epoch 5/5
42/42 [==============================] - 91s - loss: 0.6068 - acc: 0.8115 - val_loss: 4.2563 - val_acc: 0.7285

```


alexnet

```
Epoch 1/5
42/42 [==============================] - 167s - loss: 1.2400 - acc: 0.6323 - val_loss: 5.5683 - val_acc: 0.6371
Epoch 2/5
42/42 [==============================] - 162s - loss: 1.0825 - acc: 0.6441 - val_loss: 5.7694 - val_acc: 0.6371
Epoch 3/5
42/42 [==============================] - 159s - loss: 1.1416 - acc: 0.6607 - val_loss: 4.2303 - val_acc: 0.6893
Epoch 4/5
42/42 [==============================] - 159s - loss: 0.9723 - acc: 0.6501 - val_loss: 5.5769 - val_acc: 0.6371
Epoch 5/5
42/42 [==============================] - 165s - loss: 0.9243 - acc: 0.6544 - val_loss: 4.3739 - val_acc: 0.7258

```

more data aug (range from 0.1 to 0.15), batch_size = 64, steps = cnt/batch_size + 30 and gpu test

gpu alexnet

```
Epoch 1/20
41/41 [==============================] - 34s - loss: 1.3237 - acc: 0.6310 - val_loss: 6.3830 - val_acc: 0.6031
Epoch 2/20
41/41 [==============================] - 33s - loss: 1.1246 - acc: 0.6795 - val_loss: 6.3574 - val_acc: 0.6031
Epoch 3/20
41/41 [==============================] - 33s - loss: 1.0024 - acc: 0.6852 - val_loss: 6.2926 - val_acc: 0.6031
Epoch 4/20
41/41 [==============================] - 33s - loss: 1.0614 - acc: 0.6459 - val_loss: 6.3967 - val_acc: 0.6031
Epoch 5/20
41/41 [==============================] - 33s - loss: 0.9650 - acc: 0.6856 - val_loss: 5.6855 - val_acc: 0.6031
Epoch 6/20
41/41 [==============================] - 33s - loss: 0.8640 - acc: 0.6761 - val_loss: 4.5357 - val_acc: 0.7102
Epoch 7/20
41/41 [==============================] - 30s - loss: 0.7440 - acc: 0.7458 - val_loss: 4.6328 - val_acc: 0.7050
Epoch 8/20
41/41 [==============================] - 17s - loss: 0.6373 - acc: 0.7559 - val_loss: 3.2365 - val_acc: 0.7781
Epoch 9/20
41/41 [==============================] - 20s - loss: 0.6483 - acc: 0.7788 - val_loss: 2.5988 - val_acc: 0.8381
Epoch 10/20
41/41 [==============================] - 19s - loss: 0.5493 - acc: 0.8170 - val_loss: 2.9939 - val_acc: 0.8094
Epoch 11/20
41/41 [==============================] - 20s - loss: 0.6264 - acc: 0.8139 - val_loss: 2.7879 - val_acc: 0.8251
Epoch 12/20
41/41 [==============================] - 21s - loss: 0.4878 - acc: 0.8569 - val_loss: 3.2224 - val_acc: 0.7990
Epoch 13/20
41/41 [==============================] - 19s - loss: 0.4390 - acc: 0.8591 - val_loss: 2.7985 - val_acc: 0.8225
Epoch 14/20
41/41 [==============================] - 20s - loss: 0.3820 - acc: 0.8822 - val_loss: 3.1029 - val_acc: 0.8042
Epoch 15/20
41/41 [==============================] - 19s - loss: 0.4393 - acc: 0.8613 - val_loss: 3.7875 - val_acc: 0.7650
Epoch 16/20
41/41 [==============================] - 20s - loss: 0.3233 - acc: 0.8929 - val_loss: 2.2143 - val_acc: 0.8616
Epoch 17/20
41/41 [==============================] - 19s - loss: 0.3254 - acc: 0.8904 - val_loss: 2.4475 - val_acc: 0.8460
Epoch 18/20
41/41 [==============================] - 20s - loss: 0.3021 - acc: 0.9063 - val_loss: 2.9038 - val_acc: 0.8198
Epoch 19/20
41/41 [==============================] - 20s - loss: 0.3069 - acc: 0.9105 - val_loss: 2.1463 - val_acc: 0.8668
Epoch 20/20
41/41 [==============================] - 16s - loss: 0.3011 - acc: 0.8971 - val_loss: 2.2151 - val_acc: 0.8616

```

gpu simple cnn

```
```