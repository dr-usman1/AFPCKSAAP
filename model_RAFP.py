def RAFP_model(input_dim, nb_classes):
    inputs = Input([input_dim,1])
    layer0 = Flatten()(inputs)

    layer1 = Dense(128, activation='relu', name="layer1")(layer0)
    layer1 = Dropout(0.7)(layer1)

    layer2 = Dense(128, activation='relu', name="layer2")(layer1)
    layer2 = Dropout(0.7)(layer2)

    layer3 = Dense(128, activation='relu', name="layer3")(layer2)
    layer3 = Dropout(0.7)(layer3)

    layer4 = Dense(128, activation='relu', name="layer4")(layer3)
    layer4 = Dropout(0.7)(layer4)

    layer9 = Dense(nb_classes, activation='softmax', name="layer9")(layer4)

    model = Model(inputs=inputs, output=layer9)
    # model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


def RAFP_model_Skip(input_dim, nb_classes):
    inputs = Input([input_dim,1])
    layer0 = Flatten()(inputs)
    layer1 = Dense(70, kernel_initializer='random_uniform', activation='relu')(layer0)
    layer1 = Dropout(0.7)(layer1)

    layer2 = concatenate([layer0, layer1], axis=1)
    layer2 = Dense(128, kernel_initializer='random_uniform', activation='relu')(layer2)
    layer2 = Dropout(0.7)(layer2)


    layer6 = Dense(nb_classes, activation='softmax', name="layer6")(layer2)

    model = Model(input=inputs, output=layer6)

    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    return model
