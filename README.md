# PaddleOCR-VL Recognition

This is a PaddleOCR-VL recognition script that retains only the core functionality of the VL recognition model from PaddleOCR.

Additionally, it has been adapted for fine-tuned PaddleOCR-VL models [from AI Studio](https://aistudio.baidu.com/modelsdetail/41446/intro) or [from ModelScope](https://modelscope.cn/models/megemini/PaddleOCR-VL-Receipt/summary) , allowing recognition using custom prompt_label and query parameters.

## Recognition Results Display

| Input Image | Full Information Extraction | Specific Information Extraction |
|---------|---------|---------|
| ![Receipt](./images/receipt.jpeg) | ![Full Recognition](./images/rec_full.png) | ![Partial Recognition](./images/rec_part.png) |

## Environment Setup

### Install Dependencies Using requirements.txt

```bash
pip install -r requirements.txt
```

### Manual Installation (If Needed)

```bash
python -m pip install -U "paddleocr[doc-parser]"
python -m pip install https://paddle-whl.bj.bcebos.com/nightly/cu126/safetensors/safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl
python -m pip install --force-reinstall opencv-python-headless
python -m pip install numpy==1.26.4
python -m pip install json-repair
```

## Features

1. ✅ Able to initialize vl_rec_model
2. ✅ Able to input an image and directly output recognized text results
3. ❌ Does not include document preprocessing (use_doc_preprocessor)
4. ❌ Does not include layout detection (use_layout_detection)
5. ❌ Does not include chart recognition (use_chart_recognition)
6. ❌ Does not include content formatting (format_block_content)
7. ❌ Does not include layout block merging (merge_layout_blocks)
8. ❌ Does not include markdown ignore labels (markdown_ignore_labels)
9. ❌ Does not output extra information such as cls_id, score, coordinate

## Usage

### Method 1: Command Line Usage

```bash
python paddleocr_vl_rec.py --image /path/to/your/image.jpg --model_dir /path/to/model
```

Optional Parameters:

- `--model_name`: Model name (default: PaddleOCR-VL-0.9B)
- `--model_dir`: Model directory path (optional)
- `--device`: Device (e.g., 'cpu', 'gpu:0')
- `--prompt_label`: Recognition task type (default: 'ocr')
  - Optional values: 'ocr', 'table', 'formula', 'chart'
- `--query`: Additional custom prompt (optional; if provided, appended after prompt_label)
  - Supports string format in command line
- `--max_new_tokens`: Maximum number of tokens to generate (default: 4096)
- `--return_json`: Parse result as JSON format (default: False)

### Method 2: Use as a Python Module

```python
from paddleocr_vl_rec import PaddleOCRVLRec

# Initialize recognizer
recognizer = PaddleOCRVLRec(
    model_name="PaddleOCR-VL-0.9B", # Optional, defaults to "PaddleOCR-VL-0.9B"
    model_dir="path/to/your/model",
    device="gpu:0"  # or "cpu"
)

# Recognize text in image (using prompt_label)
result_text = recognizer.predict(
    image="/path/to/your/image.jpg",
    prompt_label="ocr" # Optional, defaults to "ocr"
)

print(result_text)

# Close model
recognizer.close()
```

### Method 3: Using dict or list as query

> **Note**: This method only works with fine-tuned models. The default model cannot correctly recognize and output results in JSON format.

```python
from paddleocr_vl_rec import PaddleOCRVLRec

# Initialize recognizer
recognizer = PaddleOCRVLRec(
    model_dir="path/to/your/model"
)

# Use dict as query (will be converted to JSON string)
# Return JSON format (parsed using json_repair)
result_json = recognizer.predict(
    image="/path/to/your/image.jpg",
    query={"NAME":"", "ITEMS":[]},
    return_json=True
)
# result_json is a dictionary object
print(type(result_json))  # <class 'dict'>
print(result_json)

# Use list as query (will be converted to {"item1":"", "item2":""} format)
result_json = recognizer.predict(
    image="/path/to/your/image.jpg",
    query=["item1", "item2"],
    return_json=True
)
print(result_json)

recognizer.close()
```

## Additional Examples

### OCR Text Recognition

```bash
python paddleocr_vl_rec.py --image document.jpg --prompt_label ocr
```

### Table Recognition

```bash
python paddleocr_vl_rec.py --image table.jpg --prompt_label table
```

### Formula Recognition

```bash
python paddleocr_vl_rec.py --image formula.jpg --prompt_label formula
```

### Chart Recognition

```bash
python paddleocr_vl_rec.py --image chart.jpg --prompt_label chart
```

### Combining prompt_label and query

```bash
# query will be appended after the prompt_label's prompt text
# Final prompt: "OCR:" + "{\"ITEM\":\"\", \"AMOUNT\":\"\"}"
python paddleocr_vl_rec.py --image document.jpg --prompt_label ocr --query "{\"ITEM\":\"\", \"AMOUNT\":\"\"}"
```
