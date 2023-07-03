import json
import os
import random
import shutil

import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                     MaxPooling2D)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_annotations(annotations_file):
    """Function to load annotations"""
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    return annotations


def prepare_directories(dir_path):
    """Function to prepare directories"""
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)


def process_datasets(dataset_dir, annotations, images):
    """Function to process datasets"""
    class_dirs_dataset = {}

    for category in annotations['categories']:
        class_name = category['name']
        class_dir_dataset = os.path.join(dataset_dir, class_name)
        os.makedirs(class_dir_dataset, exist_ok=True)
        class_dirs_dataset[category['id']] = class_dir_dataset

    for image in images:
        handle_image(annotations, class_dirs_dataset, image)


def handle_image(annotations, class_dirs_dataset, image):
    """Function to handle image"""
    src_path = os.path.join('annotated_dataset', image['file_name'])
    if os.path.exists(src_path):
        annotation_id = image['id']
        annotation = next((ann for ann in annotations['annotations']
                           if ann['image_id'] == annotation_id), None)
        if annotation is not None:
            class_id = annotation['category_id']
            dst_path = os.path.join(
                class_dirs_dataset[class_id], os.path.basename(image['file_name']))
            shutil.copyfile(src_path, dst_path)


def get_data_generators():
    """Function to get data generators"""
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    return train_datagen, val_datagen


def define_model_architecture(num_categories):
    """Function to define model architecture"""
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_categories, activation='softmax'))

    return model


def learning_rate_schedule(epoch):
    """Function for learning rate schedule"""
    learning_rate = 0.0001
    if epoch < 5:
        return learning_rate
    else:
        return learning_rate * tf.math.exp(-0.1)


def main():
    """Main processing function"""
    # Step 1: Load the annotations
    data_dir = 'annotated_dataset'
    annotations_file = f'{data_dir}/result.json'
    annotations = load_annotations(annotations_file)

    # Step 2: Prepare the training and validation datasets
    train_dir = 'train'
    val_dir = 'val'
    prepare_directories(train_dir)
    prepare_directories(val_dir)
    images = annotations['images']
    random.shuffle(images)

    train_ratio = 0.8
    train_size = int(len(images) * train_ratio)
    train_images = images[:train_size]
    val_images = images[train_size:]

    process_datasets(train_dir, annotations, train_images)
    process_datasets(val_dir, annotations, val_images)

    train_datagen, val_datagen = get_data_generators()
    model = define_model_architecture(num_categories=len(annotations['categories']))

    learning_rate = 0.0001
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
    lr_scheduler = LearningRateScheduler(learning_rate_schedule)

    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')

    val_generator = val_datagen.flow_from_directory(
        val_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')

    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=30,
        validation_data=val_generator,
        validation_steps=val_generator.samples // val_generator.batch_size,
        callbacks=[checkpoint, lr_scheduler])

    val_generator = val_datagen.flow_from_directory(
        val_dir, target_size=(224, 224), batch_size=1, class_mode='categorical', shuffle=False)

    y_true = val_generator.classes
    y_pred = model.predict(val_generator)
    y_pred = tf.argmax(y_pred, axis=1).numpy()

    accuracy = sum(y_true == y_pred) / len(y_true)
    print(f'Validation Accuracy: {accuracy}')

    model.save('trained_model.h5')


if __name__ == "__main__":
    main()
