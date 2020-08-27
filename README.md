# Pokemon-Classifier-V3

### Dataset: https://www.kaggle.com/abcsds/pokemon

Used Keras to train a DNN on real life data. The model can then classify which type of Pokemon it belongs to (only 3 type here) based on their six attributes.

Record of the change of the structure and performance analysis are following.

### Data pre-processing
- Derived six attributes(HP, Attack, Defense, Sp. Attack, Sp. Defense, Speed )
- Only derived three type of Pokemon(Grass, Fire, Water) since the training set is too small, it's way too complicated to do the classifier for all types
- labeled Grass as [1,0,0], Fire as [0,1,0], Water as [0,0,1]

### First trial
Since the training data is only a few (187 training set, 24 validation set) after the validation, I set the neuron for each layer to be ***1000*** and epoch to be ***150***. I found out that the batch-size performed relatively better when it's ***25***.

~~~
model.add(Dense(input_dim=6, units=1000, activation='sigmoid'))
model.add(Dense(units=1000, activation='sigmoid'))
model.add(Dense(units=3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.summary()

history = model.fit(x_train, y_train, batch_size=25, epochs=150, validation_split = 0.2)
~~~
<img src="/Users/yuwenchen/Desktop/PCV3/loss1.png" width="250"> <img src="/Users/yuwenchen/Desktop/PCV3/acc1.png" width="250">

### Second trial

The performance on training set is not sufficient enough in the first trial, therefore I increased the amount of epoch to ***300*** and change the activation function of first layer to ***relu*** to avoid gradient vanishing problem.

~~~
model.add(Dense(input_dim=6, units=1000, activation='relu'))
model.add(Dense(units=1000, activation='sigmoid'))
model.add(Dense(units=3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.summary()

history = model.fit(x_train, y_train, batch_size=25, epochs=300, validation_split = 0.2)
~~~
<img src="/Users/yuwenchen/Desktop/PCV3/loss2.png" width="250"> <img src="/Users/yuwenchen/Desktop/PCV3/acc2.png" width="250">

Turned out the performance on training set did increase (accuracy is almost 1) and the model approximately converged at 200 epoch. Still the performance on validation set is not well.

### Third trial

- **Added another layer with activation function relu to make it "deeper"...**
- **Decreased the amount of neuron to test whether it still performs well. It turned out the performance increased and the accuracy reached 1 at 100 epoch which is less than the second trial.**

~~~
model.add(Dense(input_dim=6, units=500, activation='relu'))
model.add(Dense(input_dim=6, units=500, activation='relu'))
model.add(Dense(units=1000, activation='sigmoid'))
model.add(Dense(units=3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.summary()

history = model.fit(x_train, y_train, batch_size=20, epochs=epochs, validation_split = 0.2)
~~~
 <img src="/Users/yuwenchen/Desktop/PCV3/loss3.png" width="250"> <img src="/Users/yuwenchen/Desktop/PCV3/acc3.png" width="250">

### Forth trial

Since the performance on training set is good but not testing set, I assume the overfitting happened and applied ***Dropout***.

- **Replaced the activation function of first layer to Maxout since relu can be simulated by Maxout**
- **Performance would be better if the model is almost linear when we are applying Dropout, hence I also applied Dropout for the first layer**
- ***Removed a hidden layer***

~~~
model.add(MaxoutDense(512, nb_feature=3, input_dim=6))
model.add(Dropout(0.5))
model.add(Dense(units=1024, activation='sigmoid'))
model.add(Dense(units=3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.summary()

history = model.fit(x_train, y_train, batch_size=25, epochs=epochs, validation_split = 0.2)
~~~
<img src="/Users/yuwenchen/Desktop/PCV3/loss4.png" width="250"> <img src="/Users/yuwenchen/Desktop/PCV3/acc4.png" width="250">

It's normal that the performance on training set drops after applying Dropout. The accuracy on validation set increased up to 0.78.
Also the loss on validation set was no longer increasing but decreasing gradually.
