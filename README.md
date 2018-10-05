# ImageClassification

Image Classification Model implemented by ModelZoo.


## Installation

Firstly you need to clone this repository and install dependencies with pip:

```
pip3 install -r requirements.txt
```

## Dataset

We use Fashion Mnist dataset for example.

## Usage

We can run this model like this:

```
python3 train.py
```

Outputs like this:

```
Epoch 1/20
1874/1875 [============================>.] - ETA: 0s - loss: 0.4318 - acc: 0.8427
1875/1875 [==============================] - 80s 43ms/step - loss: 0.4318 - acc: 0.8427 - val_loss: 0.3753 - val_acc: 0.8644
Epoch 2/20
1873/1875 [============================>.] - ETA: 0s - loss: 0.3295 - acc: 0.8777
Epoch 00002: saving model to checkpoints/model.ckpt
1875/1875 [==============================] - 82s 44ms/step - loss: 0.3295 - acc: 0.8777 - val_loss: 0.3684 - val_acc: 0.8716
Epoch 3/20
1872/1875 [============================>.] - ETA: 0s - loss: 0.2982 - acc: 0.8887
1875/1875 [==============================] - 70s 37ms/step - loss: 0.2984 - acc: 0.8887 - val_loss: 0.3563 - val_acc: 0.8726
Epoch 4/20
1873/1875 [============================>.] - ETA: 0s - loss: 0.2872 - acc: 0.8952
Epoch 00004: saving model to checkpoints/model.ckpt
1875/1875 [==============================] - 53s 28ms/step - loss: 0.2873 - acc: 0.8952 - val_loss: 0.3418 - val_acc: 0.8775
Epoch 5/20
1872/1875 [============================>.] - ETA: 0s - loss: 0.2679 - acc: 0.9000
1875/1875 [==============================] - 61s 33ms/step - loss: 0.2678 - acc: 0.9000 - val_loss: 0.3331 - val_acc: 0.8831
```

OK, we've finished training. Just so quickly.

## License

MIT