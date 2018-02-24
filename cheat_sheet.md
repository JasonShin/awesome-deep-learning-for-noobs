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

### Sentiment Analysis predit new text
```
text = numpy.array([‘this is an excellent sentence’])
#print(text.shape)
tk = keras.preprocessing.text.Tokenizer( nb_words=2000, lower=True,split=” “)
tk.fit_on_texts(text)
prediction = model.predict(numpy.array(tk.texts_to_sequences(text)))
print(prediction)
```
