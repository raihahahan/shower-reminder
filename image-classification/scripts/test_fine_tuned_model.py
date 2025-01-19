import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.classify_image import classify_image

def test_model(test_dir="image-classification/dataset/val"):
    """
    Tests the fine-tuned model on a validation dataset.

    Args:
        test_dir (str): Path to the validation dataset directory.

    Prints:
        Detailed metrics for each category including precision, recall, and F1 score.
    """
    if not os.path.exists(test_dir):
        raise ValueError(f"Test directory '{test_dir}' does not exist!")

    # Initialize metrics for each category
    metrics = {
        "showerhead": {"true_pos": 0, "false_pos": 0, "false_neg": 0, "total": 0},
        "not_showerhead": {"true_pos": 0, "false_pos": 0, "false_neg": 0, "total": 0}
    }

    # Process each category
    for category in os.listdir(test_dir):
        category_path = os.path.join(test_dir, category)
        if os.path.isdir(category_path):
            print(f"\nTesting category: {category}")
            
            for image_file in os.listdir(category_path):
                image_path = os.path.join(category_path, image_file)

                try:
                    predicted_label, confidence = classify_image(image_path)
                    metrics[category]["total"] += 1

                    # Update metrics
                    if predicted_label == category:
                        metrics[category]["true_pos"] += 1
                    else:
                        metrics[category]["false_neg"] += 1
                        metrics[predicted_label]["false_pos"] += 1

                    print(
                        f"Image: {image_file}, Predicted: {predicted_label}, "
                        f"Confidence: {confidence:.2f}, Correct: {predicted_label == category}"
                    )
                except Exception as e:
                    print(f"Error processing image {image_path}: {e}")

    # Print detailed metrics for each category
    print("\n=== Detailed Results ===")
    total_images = sum(m["total"] for m in metrics.values())
    total_correct = sum(m["true_pos"] for m in metrics.values())

    for category, m in metrics.items():
        true_pos = m["true_pos"]
        false_pos = m["false_pos"]
        false_neg = m["false_neg"]
        total = m["total"]

        # Calculate metrics
        accuracy = true_pos / total if total > 0 else 0
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"\nMetrics for {category}:")
        print(f"Total images: {total}")
        print(f"True Positives: {true_pos}")
        print(f"False Positives: {false_pos}")
        print(f"False Negatives: {false_neg}")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Precision: {precision:.2%}")
        print(f"Recall: {recall:.2%}")
        print(f"F1 Score: {f1:.2%}")

    # Print overall accuracy
    print(f"\nOverall Accuracy: {total_correct/total_images:.2%}")

if __name__ == "__main__":
    test_model()
