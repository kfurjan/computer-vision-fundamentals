import argparse
import cv2
import json
import numpy as np
import tensorflow as tf


def create_arg_parser():
    """Create and return an ArgumentParser object."""
    parser = argparse.ArgumentParser(description='Perform object detection for a given image')
    parser.add_argument('image_path',
                        metavar='image_path',
                        type=str,
                        help='Path to the image')
    return parser


def load_input_image(image_path):
    """Load and return input image."""
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (224, 224))
    image_normalized = image_resized / 255.0
    return np.expand_dims(image_normalized, axis=0), image


def draw_bounding_box_on_image(image, class_name, accuracy):
    """Draw bounding box and display the detected class and accuracy in it."""
    # Image dimensions
    image_height, image_width, _ = image.shape
    # Bounding box coordinates
    x, y, width, height = 10, 10, image_width - 20, image_height - 20
    # Draw bounding box
    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
    # Display class and accuracy
    text = f'{class_name}: {accuracy:.2f}'
    cv2.putText(image, text, (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


def display_image(image, object_detected):
    """Display image with or without bounding box."""
    if not object_detected:
        cv2.putText(image, 'No Object Detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.imshow('Object Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    # Parse input arguments
    parser = create_arg_parser()
    args = parser.parse_args()

    # Load image and model
    input_image, image = load_input_image(args.image_path)
    model = tf.keras.models.load_model('trained_model.h5')

    # Load annotations and class labels
    with open('annotated_dataset/result.json', 'r') as f:
        annotations = json.load(f)
    class_labels = annotations['categories']

    # Perform object detection
    predictions = model.predict(input_image)
    predicted_labels = np.argmax(predictions, axis=1)

    # Flag to indicate if any object is detected
    object_detected = False

    # Iterate over all predicted labels and their corresponding annotations
    for label_id, annotation in zip(predicted_labels, annotations['annotations']):
        class_name = class_labels[label_id]['name']
        accuracy = predictions[0][label_id]

        print(f'Detected Object: {class_name}, Accuracy: {accuracy:.2f}')

        object_detected = True

        # Draw bounding box and display the detected class and accuracy
        draw_bounding_box_on_image(image, class_name, accuracy)

    # Display the image with or without bounding box
    display_image(image, object_detected)


if __name__ == "__main__":
    main()
