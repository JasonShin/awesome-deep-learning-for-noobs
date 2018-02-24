## Numpy

### Getting number total of words
```
len(numpy.unique(numpy.hstack(X)))
```

### Finding number of classes
```
np.unique(y)
```

### Mean and standard deviation
```
len(numpy.unique(numpy.hstack(X)))
```

### Predictable numpy random
You would fixate random for reproducibility
```
seed = 7
numpy.ramdom.seed(seed)
```

## Keras Data Processing

### Get model summary
```
model.summary()
```

### Truncate or Pad a dataset
```
import keras.preprocessing import sequence
sequence.pad_sequences(X_train, maxlen=max_words)
```
