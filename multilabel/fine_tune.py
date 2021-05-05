from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def fine_tune_pretrained_model(base_model, df_train):

    # linear probe
    linear_probe = Dense(14, activation="sigmoid")

    # model
    x_in = base_model.inputs
    x = base_model(x_in)
    x_out = linear_probe(x)

    model = Model(inputs=x_in, outputs=x_out)

    # train and validation generators
    datagen = ImageDataGenerator(
        rescale=1./255.,
        validation_split=0.1)

    generator_train = datagen.flow_from_dataframe(
        df_train,
        directory="data/chest_x_rays",
        x_xol="filename",
        y_col="labels",
        class_mode="categorical",
        target_size=(299,299),
        batch_size=32,
        subset="training")

    generator_val = datagen.flow_from_dataframe(
        df_train,
        directory="data/chest_x_rays",
        x_xol="filename",
        y_col="labels",
        class_mode="categorical",
        target_size=(299,299),
        batch_size=32,
        subset="validation")

    # train only the linear probe
    base_model.trainable = False

    # compile model
    model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=[AUC(multi_label=True), Precision(), Recall()])

    # train model
    model.fit(
        generator_train,
        steps_per_epoch=generator_train.n // generator_train.batch_size,
        validation_data=generator_val,
        validation_steps=generator_val.n // generator_val.batch_size,
        epochs=2,
        callbacks=[
            ModelCheckpoint(
                filepath="data/xception_linear_probe.h5",
                verbose=1,
                save_best_only=True)])
    
    # fine tune whole model
    base_model.trainable = True

    # compile model
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="binary_crossentropy",
        metrics=[AUC(multi_label=True), Precision(), Recall()])

    # train model
    model.fit(
        generator_train,
        steps_per_epoch=generator_train.n // generator_train.batch_size,
        validation_data=generator_val,
        validation_steps=generator_val.n // generator_val.batch_size,
        epochs=20,
        callbacks=[
                   EarlyStopping(patience=5, restore_best_weights=True),
                   ModelCheckpoint(
                       filepath="data/xception_linear_probe_fine_tuned.h5",
                       verbose=1,
                       save_best_only=True),
                   ReduceLROnPlateau(patience=1, verbose=1, cooldown=2)])


if __name__ == "__main__":

    from extract_features import get_train_test_dfs

    from tensorflow.keras.applications import Xception

    df_train, _ = get_train_test_dfs()

    print(df_train)


    base_model = Xception(
        include_top=False,
        weights="imagenet",
        input_shape=(299,299,3),
        pooling="max")

    fine_tune_pretrained_model(base_model, df_train)
