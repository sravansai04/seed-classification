from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input,Dense,AveragePooling2D,Dropout,Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from tensorflow import lite
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import pandas as pd
from keras.models import save_model

DATADIR = "D:\SeedsClassification\Dataset"

CATEGORIES = ["Bad Groundnut","Bad Moongdal","Bad Pigeonpea","Good Groundnut","Good Moongdal","Good Pigeonpea"]

INIT_LR = 0.001
EPOCHS = 10
BS = 16

data = []
labels = []
dataset = [0, 0, 0, 0, 0, 0]

for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    catnum = CATEGORIES.index(category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)

        data.append(image)
        labels.append(catnum)
        dataset[CATEGORIES.index(category)] += 1

data = np.array(data, dtype="float32")
labels = np.array(labels)

dicto={}
for i in range(0,len(dataset)):
    dicto[i] = [CATEGORIES[i],dataset[i]]
dicto[i] = ["Total",sum(dataset)]
df = pd.DataFrame.from_dict(dicto,orient='index',columns=['Seed type','Image count'])
print(df)

fig, ax = plt.subplots(figsize=(6,6))
patches, texts, pcts = ax.pie(dataset, labels=CATEGORIES, autopct='%.2f%%',
    wedgeprops={'linewidth': 3.0, 'edgecolor': 'white'},
    textprops={'size': 'large'})
plt.setp(pcts, color='white', fontweight='bold')
ax.set_title('Dataset distribution', fontsize=20)
plt.tight_layout()

labels = to_categorical(labels,6)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state = 42)

aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")
baseModel = MobileNetV2(weights="imagenet", include_top=False,input_tensor=Input(shape=(224, 224, 3)))

for layer in baseModel.layers:
    layer.trainable = False

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(6, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)
for layer in model.layers:
    print(layer.output_shape)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS)
predIdxs = model.predict(testX, batch_size=BS)
predIdxs = np.argmax(predIdxs, axis=1)

print(classification_report(testY.argmax(axis=1), predIdxs,target_names=CATEGORIES))

cm_test = confusion_matrix(y_pred=predIdxs, y_true=testY.argmax(axis=1))
cm_test

ax= plt.subplot()
sns.heatmap(cm_test, annot=True,fmt="d",cmap='Blues',xticklabels=CATEGORIES, yticklabels=CATEGORIES)
ax.set_xlabel('Predicted type', fontsize=15)
ax.set_ylabel('Actual type', fontsize=15)
plt.show()

N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss",color='red')
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")

N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss",color='blue')
plt.title("Validation Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")

N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc",color='green')
plt.title("Training Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")

N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc",color='black')
plt.title("Validation Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")

data = []
path = f"D:/SeedsClassification/testdata/test13.jpg"
image = load_img(path, target_size=(224, 224))
image = img_to_array(image)
image = preprocess_input(image)
data.append(image)
data = np.array(data, dtype="float32")
lt = model.predict(data)
pred = np.argmax(lt)
print(CATEGORIES[pred])

actual = ['Good Pigeonpea', 'Good Pigeonpea', 'Good Pigeonpea', 'Good Pigeonpea', 'Good Groundnut', 'Good Groundnut',
          'Good Groundnut', 'Good Groundnut', 'Good Moongdal', 'Bad Groundnut', 'Bad Groundnut', 'Good Moongdal',
          'Bad Pigeonpea',
          'Good Moongdal', 'Bad Pigeonpea', 'Bad Pigeonpea', 'Bad Moongdal', 'Bad Moongdal', 'Bad Moongdal']
columns = ["Actual Type", "Predicted type", "Validation", "Bad Groundnut", "Bad Moongdal", "Bad Pigeonpea",
           "Good Groundnut", "Good Moongdal", "Good Pigeonpea"]
dicto = {}
predicted = []
correct = 0
incorrect = 0
for i in range(1, 20):
    data = []
    path = f"D:/SeedsClassification/testdata/test{i}.jpg"
    image = load_img(path, target_size=(224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)
    data.append(image)
    data = np.array(data, dtype="float32")
    lt = model.predict(data)
    pred = np.argmax(lt)
    comp = []
    comp.append(actual[i - 1])
    comp.append(CATEGORIES[pred])
    predicted.append(pred)
    comp.append('Correct' if CATEGORIES[pred] == actual[i - 1] else 'Incorrect')
    for j in range(0, len(lt[0])):
        comp.append("{:.2f}".format(lt[0][j] * 100))
    dicto[i - 1] = comp

df = pd.DataFrame.from_dict(dicto, orient='index', columns=columns)
print(df)

for i in range(0,len(actual)):
    actual[i] = int(CATEGORIES.index(actual[i]))
actual
actual = to_categorical(actual,6)
cr = classification_report(actual.argmax(axis=1), predicted,target_names=CATEGORIES)
print(cr)

model.save("seedmodel.h5")

converter = lite.TFLiteConverter.from_keras_model(model)
tfmodel = converter.convert()
open('seedmodel.tflite','wb').write(tfmodel)