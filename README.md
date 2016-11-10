## Udacity Deep Learning Assignments
* Assignments for udacity deep learning class.
* Do it yourself, Compare it with your colleague.
* I like OpenCV better than matplotlib.
* I prefer using '.py' to '.ipynb'. 

## Notice
* People who are familar with python (like me), There are .py files instead of .ipynb in each folder

### *Converted with below comman line*
```linux
jupyter nbconvert --to python *.ipynb
```

## Assignment Progress
|Assignment|# of Problems|Done|Valid Accuracy|Test Accuracy|
|---|---:|---:|---:|---:|
|1_notmnist|6|6|75.9%|84.0%|
|2_fullyconnected|1|1|79.5%|87.0%|
|3_regularization|4|4|90.9%|96.3%|
|4_convolutions|2|2|88.8%|95.0%|
|5_word2vec|1||||
|6_lstm|3||||

## Version
* Tensorflow '0.11.0rc0'
* OpenCV '2.4.8'

### *Version check with below python code*
```linux
import tensorflow as tf
print(tf.__version__)
import cv2
print(cv2.__version__)
```
