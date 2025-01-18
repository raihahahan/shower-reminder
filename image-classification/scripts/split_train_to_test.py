import os
import shutil
import random

def split_train_to_test(train_dir, test_dir, split_ratio=0.2):
    if not os.path.exists(train_dir):
        raise ValueError(f"Training directory '{train_dir}' does not exist.")

    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    for category in os.listdir(train_dir):
        category_path = os.path.join(train_dir, category)

        if not os.path.isdir(category_path):
            print(f"Skipping non-directory item: {category_path}")
            continue

        # Debug: Print category being processed
        print(f"Processing category: {category}")

        # Create the corresponding category directory in the test set
        test_category_path = os.path.join(test_dir, category)
        os.makedirs(test_category_path, exist_ok=True)

        # Get all image files in the category
        images = [f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]
        print(f"Found {len(images)} images in category '{category}'")

        if len(images) == 0:
            print(f"No images found in category '{category}'. Skipping.")
            continue

        # Shuffle and select a subset for the test set
        num_test_images = int(len(images) * split_ratio)
        test_images = random.sample(images, num_test_images)
        print(f"Moving {num_test_images} images to test directory for category '{category}'")

        for image in test_images:
            src_path = os.path.join(category_path, image)
            dest_path = os.path.join(test_category_path, image)

            # Move the image to the test set
            shutil.move(src_path, dest_path)

        print(f"Moved {num_test_images} images from '{category_path}' to '{test_category_path}'.")

if __name__ == "__main__":
    train_dir = "dataset/train"  # Path to your training dataset
    test_dir = "dataset/val"    # Path to your test dataset
    split_ratio = 0.2             # Proportion of training images to move to the test set

    split_train_to_test(train_dir, test_dir, split_ratio)
