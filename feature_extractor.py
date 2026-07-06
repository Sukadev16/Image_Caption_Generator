from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

import numpy as np
import os
import pickle


# Load pretrained ResNet50

base_model = ResNet50(weights='imagenet')

# Remove final classification layer

model = Model(
    inputs=base_model.inputs,
    outputs=base_model.layers[-2].output
)


def extract_features(folder_path):

    features = {}

    for img_name in os.listdir(folder_path):

        # Process only jpg images

        if not img_name.lower().endswith(".jpg"):
            continue

        img_path = os.path.join(folder_path, img_name)

        try:

            img = image.load_img(
                img_path,
                target_size=(224, 224)
            )

            img_array = image.img_to_array(img)

            img_array = np.expand_dims(
                img_array,
                axis=0
            )

            img_array = preprocess_input(img_array)

            feature = model.predict(
                img_array,
                verbose=0
            )

            features[img_name] = feature

            print("Processed:", img_name)

        except Exception as e:

            print("Error:", img_name)
            print(e)

            continue

    return features


if __name__ == "__main__":

    # Images are inside data/Images

    image_folder = "data/Images"

    features = extract_features(image_folder)

    print("\nTotal Features :", len(features))

    first_key = list(features.keys())[0]

    print(
        "Feature Shape :",
        features[first_key].shape
    )

    with open("features.pkl", "wb") as f:

        pickle.dump(features, f)

    print("\nfeatures.pkl saved successfully")