# Amazon ML Challenge 2024
## Problem Statement : Feature Extraction from Images

## Overview
The goal is to create a machine learning model that extracts entity values from images. This is particularly useful in industries such as healthcare, e-commerce, and content moderation, where obtaining precise product information directly from images is essential. The extracted information includes weight, volume, voltage, wattage, dimensions, and more, which are critical for digital marketplaces.

## Dataset Description
The dataset consists of the following columns:
- **index**: Unique identifier for the data sample.
- **image_link**: URL to download the product image. Example: `https://m.media-amazon.com/images/I/71XfHPR36-L.jpg`.
  - To download images, use the `download_images` function in `src/utils.py`. See sample usage in `src/test.ipynb`.
- **group_id**: Category code of the product.
- **entity_name**: Name of the product entity (e.g., `item_weight`).
- **entity_value**: Value associated with the entity (e.g., `34 gram`).
  - Note: `entity_value` is not present in `test.csv` as it is the target variable.

## Methodology
![Screenshot 2025-03-04 at 8 12 31 PM](https://github.com/user-attachments/assets/194fc944-02cc-4b15-a904-56ea0fb5d91c)

Our approach involves:
1. **Data Preprocessing**:
   - Download and clean images from `image_link`.
   - Convert images to a standardized format (e.g., resizing, grayscale, or RGB normalization).
   - Augment data where necessary for better generalization.

2. **Feature Extraction**:
   - Apply Optical Character Recognition (OCR) using **EasyOCR** to detect and extract textual information from images.

3. **Entity Recognition & Value Prediction**:
   - Train a Named Entity Recognition (NER) model using **BERT-based transformers** to classify extracted text into predefined entities.
   - Utilize a regression model or sequence-to-sequence transformer to predict entity values from extracted text and features.

4. **Post-processing & Unit Normalization**:
   - Map extracted values to allowed units defined in `constants.py`.
   - Apply rule-based transformations to ensure consistency in formatting (e.g., `2 kilogram` instead of `2kg`).

5. **Model Training & Evaluation**:
   - Train the model on `train.csv` using **cross-validation**.
   - Evaluate predictions using **F1-score**, ensuring accurate extraction and classification.
   - Fine-tune hyperparameters and optimize inference time for better performance.

6. **Output Formatting & Validation**:
   - Ensure outputs match the required format in `sample_test_out.csv`.
   - Validate predictions using `sanity.py` to check for formatting correctness before submission.

## Output Format
The model should generate a CSV file with the following columns:
- **index**: Unique identifier of the data sample.
- **prediction**: A string formatted as `x unit`, where `x` is a float and `unit` is one of the allowed units.
  - Example: `2 gram`, `12.5 centimetre`, `2.56 ounce`.
  - Invalid cases: `2 gms`, `60 ounce/1.7 kilogram`, `2.2e2 kilogram`.
- Ensure that all indices from `test.csv` are included in the output.
- If no value is found, return an empty string (`""`).

## Project Structure
### Source Files
- `src/sanity.py`: Ensures the final output file passes formatting checks. 
- `src/utils.py`: Contains helper functions, including downloading images.
- `src/constants.py`: Defines allowed units for each entity type.
- `sample_code.py`: A dummy script for generating a properly formatted output file (optional usage).

### Dataset Files
- `dataset/train.csv`: Training data with labels (`entity_value`).
- `dataset/test.csv`: Test data without labels.
- `dataset/sample_test.csv`: Sample test input.
- `dataset/sample_test_out.csv`: Sample formatted output file.

## Constraints
- Format the output exactly as `sample_test_out.csv`.
- Run the output through `src/sanity.py` to ensure formatting correctness.
- Use only the allowed units specified in `src/constants.py`.

## Evaluation Criteria
Submissions will be evaluated using the **F1-score**, based on the following classification:
- **True Positives (TP)**: `OUT != ""` and `GT != ""` and `OUT == GT`
- **False Positives (FP)**: `OUT != ""` and `GT != ""` and `OUT != GT`
- **False Positives (FP)**: `OUT != ""` and `GT == ""`
- **False Negatives (FN)**: `OUT == ""` and `GT != ""`
- **True Negatives (TN)**: `OUT == ""` and `GT == ""`

### F1-score Calculation
\[ F1 = \frac{2 \times Precision \times Recall}{Precision + Recall} \]
where:
- **Precision** = \( \frac{TP}{TP + FP} \)
- **Recall** = \( \frac{TP}{TP + FN} \)

## Appendix: Allowed Units
A mapping of entity types to their allowed units:
```python
entity_unit_map = {
  "width": {"centimetre", "foot", "millimetre", "metre", "inch", "yard"},
  "depth": {"centimetre", "foot", "millimetre", "metre", "inch", "yard"},
  "height": {"centimetre", "foot", "millimetre", "metre", "inch", "yard"},
  "item_weight": {"milligram", "kilogram", "microgram", "gram", "ounce", "ton", "pound"},
  "maximum_weight_recommendation": {"milligram", "kilogram", "microgram", "gram", "ounce", "ton", "pound"},
  "voltage": {"millivolt", "kilovolt", "volt"},
  "wattage": {"kilowatt", "watt"},
  "item_volume": {"cubic foot", "microlitre", "cup", "fluid ounce", "centilitre", "imperial gallon", "pint", "decilitre", "litre", "millilitre", "quart", "cubic inch", "gallon"}
}
```

