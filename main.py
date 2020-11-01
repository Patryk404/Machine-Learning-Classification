import tensorflow as tf

from input_fn import input_fn, input_fn_for_user

import pandas as pd

CSV_TREES_NAMES = ["dab","tree","big tree","big big tree","omg tree","mlg tree","OSHIT TREE","MLG OSHIT TREE","no tree"]

dftrain = pd.read_csv('./trees.csv') # training data

train_y = dftrain.pop("species")

for index1,tree1 in enumerate(train_y):
    for index2,tree2 in enumerate(CSV_TREES_NAMES):
        if(tree1==tree2):
            train_y[index1] = index2
train_y = train_y.astype('int64') #WE MUST DO the conversion to int after that
print(dftrain)

# Feature columns describe how to use the input.
my_feature_columns = []
for key in dftrain.keys(): # returning our columns(train.keys())
    my_feature_columns.append(tf.feature_column.numeric_column(key=key, dtype=tf.float32))

classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # Two hidden layers of 30 and 10 nodes respectively.
    hidden_units=[30, 10],
    # The model must choose between 9 classes of TREES.
    n_classes=9)

classifier.train(input_fn = lambda: input_fn(dftrain,train_y,training=True),steps=5000)

features = ['Girth','Height','Volume']
predict = {} 
print("Please enter numeric values")
for feature in features:
    valid = True
    while valid: 
        val = input(feature + ": ")
        if not val.isdigit(): valid = False

    predict[feature] = [float(val)]

predictions = classifier.predict(input_fn=lambda: input_fn_for_user(predict))
for pred_dict in predictions:
    print(pred_dict['class_ids'])
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]
    print('Prediction is "{}" ({:.1f}%)'.format(
        CSV_TREES_NAMES[class_id], 100 * probability))


## Classification works only on numbers like regression lol