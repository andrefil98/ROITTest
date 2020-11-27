import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt

import os
import random
import gc

veri = 0
diretorio = "./TesteROIT"
if 'modelo.h5' in os.listdir(diretorio):
    veri = 1

Epocas = 3
columns = 5
TotalCachorrosTreino = 12500
TotalGatosTreino = 12500

if veri == 0:
    train_dir_cat = './dataset_treino/Dog'
    train_dir_dog = './dataset_treino/Cat'
    train_dogs = ['./dataset_treino/Dog/{}'.format(i) for i in os.listdir(train_dir_cat)]  # Pega as imagem de cachorro
    train_cats = ['./dataset_treino/Cat/{}'.format(i) for i in os.listdir(train_dir_dog)]  # Pega as imagens de gato
    train_imgs = train_dogs[:TotalCachorrosTreino] + train_cats[:TotalGatosTreino]
    random.shuffle(train_imgs)  # randomiza
    print(len(train_cats))
    print(len(train_dogs))
    del train_dogs
    del train_cats
    gc.collect()
    print(len(train_imgs))
test_dir = './dataset_teste_api/'
test_imgs = ['./dataset_teste_api/{}'.format(i) for i in os.listdir(test_dir)]  # Pega as imagens de teste

print(len(test_imgs))


# junta os dois

random.shuffle(test_imgs) # randomiza

# print(train_imgs)
# Limpa a lista


# tamanho da imagem
Image_Width = 150
Image_Height = 150
Image_Size = (Image_Width, Image_Height)
Image_Channels = 3



def read_and_process_image(list_of_images):
    X = []  # images
    y = []  # labels
    for image in list_of_images:
        print(image)
        procura = image
        p = load_img(image, target_size=(150, 150))
        p = img_to_array(p)
        X.append(p)
    # Pega  se vai ser cachorro ou gato
        if 'Dog' in procura:
            y.append(1)
        elif 'Cat' in procura:
            y.append(0)
    print(len(X))
    print(len(y))
    return X, y
if veri == 0:
    X, y = read_and_process_image(train_imgs)

    import seaborn as sns
    del train_imgs
    gc.collect()

    #Converte a lista para numpy array
    X = np.array(X)
    y = np.array(y)


    sns.countplot(y)
    plt.title('Rótulos para gatos e cachorros:')

    print("Formato(shape) das imagens de treino:", X.shape)
    print("Formato(shape) dos rótulos          :", y.shape)

    #Divide o treino e o teste
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=2)

    print("Shape of train images is:", X_train.shape)
    print("Shape of validation images is:", X_val.shape)
    print("Shape of labels is:", y_train.shape)
    print("Shape of labels is:", y_val.shape)


    #limpa a memoria
    del X
    del y
    gc.collect()

    #pega comprimento do traino and validação dados
    ntrain = len(X_train)
    nval = len(X_val)

    batch_size = 64

    # inicializando a rede neural
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, \
        Dropout, Flatten, Dense, Activation, \
        BatchNormalization



    from keras import optimizers
    from keras.preprocessing.image import ImageDataGenerator

    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(Image_Width, Image_Height, Image_Channels)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop', metrics=['accuracy'])

    # teste do modelo

    model.summary()

    model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])



#Cria a configuração de argumento
    train_datagen = ImageDataGenerator(rescale=1./255,   #Escala da imagem vai ser entre 0 e 1
                                    rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,)

    val_datagen = ImageDataGenerator(rescale=1./255)


#Creador de gerador de imagem
    train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
    val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> INICIO da Parte de Treinamento
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

#15 min por epoca




if veri == 0:

    history = model.fit(train_generator,
                           steps_per_epoch=ntrain // batch_size,
                           epochs=Epocas,
                           validation_data=val_generator,
                           validation_steps=nval // batch_size)
    model.save('modelo.h5')

    #plota a curva e o valor do train
    #Pega os detailes form do history object
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    #   Traino e validatição precisão
    plt.plot(epochs, acc, 'b', label='Training accurarcy')
    plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
    plt.title('Training and Validation accurarcy')
    plt.legend()
    plt.figure()
    #Valores de traino e validação perdidos
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.legend()
    plt.show()

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> FIM da Parte de Treinamento
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

from keras.models import load_model
model=load_model('modelo.h5')
from flask import Flask , request
import base64
app = Flask("teste")

@app.route("/pegaimagem",methods=["POST"])
def pegaimagem():
    body = request.get_json()
    imgdata = base64.b64decode(body["img"])
    filename = './dataset_teste_api/0.jpg'
    with open(filename, 'wb') as f:
        f.write(imgdata)
    cachorro =0
    gato = 0
    test_dir = './dataset_teste_api/'
    test_imgs = ['./dataset_teste_api/{}'.format(i) for i in os.listdir(test_dir)]
    ImagensParaAvaliar = 1
    X_test, y_test = read_and_process_image(test_imgs[0:ImagensParaAvaliar])
    x = np.array(X_test)
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    i = 0
    text_labels = []
    plt.figure(figsize=(20, 20))

    for batch in test_datagen.flow(x, batch_size=1):
        pred = model.predict(batch)
        pred1 = np.max(pred)
        pred2 = np.min(pred)
        pred = pred1 - pred2
        print(pred)
        if pred > 0.50:
            cachorro = 1
            text_labels.append(f'Cachorro {pred:.2f}')
        else:
            gato = 1
            text_labels.append(f'Gato {pred:.2f}')
        # Número de linhas, número de colunas
        plt.subplot((ImagensParaAvaliar / columns) + 1, columns, i + 1)
        plt.title('' + text_labels[i])
        imgplot = plt.imshow(batch[0])
        i += 1
        if i % ImagensParaAvaliar == 0:
            break
    plt.show()
    if cachorro == 1:
        return "Cachorro"
    if gato == 1:
        return "Gato"
app.run()



