from keras.models import Sequential
from keras.layers import Dense

import numpy as np

seed = 7
np.random.seed(seed)

dataset = np.loadtxt("pima_indians_diabetes.csv", delimiter=',')

# split the dataset, between training set, X...
# we use conventional slicing
X = dataset[:,:8]
# ... and target y
y = dataset[:,8]

# cite: Models in Keras are defined as a sequence of layers.
# cite: In this example, we will use a fully-connected network structure with three layers.
# Weights are initialized at random between (0, 0.05) drawn from the Uniform : weights ~ Uniform(0.05)
# the node activation function is the relu (linear rectifier), and a sigmoid for the output.
# first layer has 12 nodes, the second 8 and a perceptron at the end.

# create model
print("Create model...")
model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

# Compile model
print("Compiling model...")
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
print("Training model...")
model.fit(X, y, nb_epoch=150, batch_size=10)

# Evaluate
print("Training ended...")
scores = model.evaluate(X, y)
print("{0}: {1} %".format(model.metrics_names[1], scores[1]*100))