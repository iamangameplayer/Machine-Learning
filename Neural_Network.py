from tensorflow import keras

mnist=keras.datasets.mnist
(X_train , Y_train),(X_test , Y_test) = mnist.load_data()
X_train , X_test = X_train/255.0 , X_test/255.0

model = keras.Sequential(
    [keras.layers.Input(shape=(28,28)),
     keras.layers.Flatten(),
     keras.layers.Dense(128 , activation='relu'),
     keras.layers.Dense(10 , activation='softmax')
        ])

model.compile(optimizer='adam' , loss='sparse_categorical_crossentropy' , metrics=['accuracy'])

model.fit(X_train , Y_train , epochs= 5 , verbose=1)

test_loss , test_acc = model.evaluate(X_test , Y_test)
print(f"Test Accuracy {test_acc}")

pred=model.predict(X_test[:5])
print(f"Prediction for the frist 5 Images {pred.argmax(axis=1)}")

model.summary()

