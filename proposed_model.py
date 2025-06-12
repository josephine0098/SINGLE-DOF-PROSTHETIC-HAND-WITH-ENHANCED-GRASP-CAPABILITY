import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv1D, PReLU, BatchNormalization,
                                     DepthwiseConv2D, Concatenate, MaxPooling1D,
                                     LSTM, GRU, Dense, Reshape, Flatten)
from tensorflow.keras.models import Model


def inception_module(x):
    path1 = Conv1D(64, 1, padding='same', activation='relu')(x)

    path2 = Conv1D(64, 1, padding='same', activation='relu')(x)
    path2 = Conv1D(64, 3, padding='same', activation='relu')(path2)

    path3 = Conv1D(64, 1, padding='same', activation='relu')(x)
    path3 = Conv1D(64, 5, padding='same', activation='relu')(path3)

    path4 = MaxPooling1D(pool_size=3, strides=1, padding='same')(x)
    path4 = Conv1D(64, 1, padding='same', activation='relu')(path4)

    return Concatenate()([path1, path2, path3, path4])


def temporal_conv_block(x):
    x = Conv1D(64, 1, padding='same')(x)
    x = PReLU()(x)
    x = BatchNormalization()(x)
    x = DepthwiseConv2D(kernel_size=(3, 1), padding='same')(tf.expand_dims(x, axis=2))
    return tf.squeeze(x, axis=2)


def proposed_model(xtrain, ytrain, xtest, ytest):
    input_layer = Input(shape=(xtrain.shape[1], xtrain.shape[2]))

    # TCN path
    tcn = Conv1D(64, 1, padding='same')(input_layer)
    tcn = PReLU()(tcn)
    tcn = BatchNormalization()(tcn)
    tcn = temporal_conv_block(tcn)

    # Skip connection
    skip = Conv1D(64, 1, padding='same')(input_layer)
    skip = BatchNormalization()(skip)
    skip = PReLU()(skip)
    combined = Concatenate()([skip, tcn])

    # GoogleNet (Inception-style) module
    inception_out = inception_module(combined)

    # Feature stitching and GRU-LSTM
    lstm_out = LSTM(64, return_sequences=False)(inception_out)
    gru_out = GRU(64, return_sequences=False)(inception_out)
    rnn_combined = Concatenate()([lstm_out, gru_out])

    # Meta-learner
    x = Dense(64, activation='relu')(rnn_combined)
    x = Dense(32, activation='relu')(x)
    output = Dense(ytrain.shape[1], activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(xtrain, ytrain, epochs=50, batch_size=32, validation_data=(xtest, ytest), verbose=1)

    ypred = model.predict(xtest)
    return ypred
