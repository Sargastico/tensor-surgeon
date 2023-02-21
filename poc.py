from keras.models import Sequential, model_from_json
from keras.layers import Dense
import surgeon
import numpy

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")

# split into input (X) and output (Y) variables
X = dataset[:, 0:8]
Y = dataset[:, 8]

# create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu', name="input"))
model.add(Dense(8, activation='relu', name="mid1"))
model.add(Dense(8, activation='relu', name="mid2"))
model.add(Dense(8, activation='relu', name="mid3"))
model.add(Dense(1, activation='sigmoid', name="output"))

model.summary()

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, epochs=400, batch_size=10, verbose=0)

# evaluate the model
scores = model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# serialize model to YAML
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# load JSON and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

# Find custom layers that are not pure Keras layers/objects, but brought directly by Tensorflow backend.
custom_layers = {}
for layer in loaded_model.layers:
  if "relu6" in layer.name or "swish" in layer.name:
    custom_layers[layer.name] = layer

# Get head and tail models after apllying split_network. Arguments are explained. Make sure you have read its document.
head, tail = surgeon.split_network(model=loaded_model,
                                     split_layer_name="mid2",
                                     on_head=True,
                                     names=("head_model", "tail_model"),
                                     custom_objects=custom_layers)

print("\n========= MODEL SPLIT RESULT =========\n")

# Do inference by head model
head_pred = head.predict(X)
print("Output shape of the head model: ", head_pred.shape)

# Do inference by tail model
tail_pred = tail.predict(head_pred)
print("Output shape of the tail model: ", tail_pred.shape)

print("%.2f%%" % (score[1]*100))