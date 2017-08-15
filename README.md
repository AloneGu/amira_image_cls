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
