# Fine-tuning PaddleOCR-VL with New Approach -- Prompt and Information Extraction

> AI Studio Project Address: [Fine-tuning PaddleOCR-VL with New Approaches -- Prompt and Information Extraction](https://aistudio.baidu.com/projectdetail/9857242), which can be run directly in AI Studio's A100 environment (V100 environment can only perform model inference, not fine-tuning)

## Introduction

PaddleOCR-VL is a vision-language model (VLM) specifically designed for document parsing, achieving SOTA performance in both page-level document parsing and element-level recognition.

![paddleocr](./images/paddleocr.png)

When using PaddleOCR-VL, you can complete various document type understanding tasks through prompts, including text recognition (OCR), table recognition, formula recognition, and chart recognition:

```python
CHOSEN_TASK = "ocr"  # Options: 'ocr' | 'table' | 'chart' | 'formula'
PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
}
```

Currently, the fine-tuning of PaddleOCR-VL [PaddleOCR-VL-0.9B SFT](https://github.com/PaddlePaddle/ERNIE/blob/release/v1.4/docs/paddleocr_vl_sft_zh.md) also revolves around these four types of tasks. For example, [Fine-tuning PaddleOCR-VL for Manga](https://pfcc.blog/posts/paddleocr-vl-for-manga) introduces how to fine-tune the model to improve recognition accuracy for manga text.

This article introduces a new fine-tuning approach, starting from fine-tuning the `prompt` of PaddleOCR-VL, and explains how to fine-tune PaddleOCR-VL for `information extraction`.

Here's an example using a receipt:

| Input Image | Before Fine-tuning (OCR Recognition) | After Fine-tuning (Extract Specific Information) |
|-------------|--------------------------------------|--------------------------------------------------|
| ![Receipt](./images/receipt.jpeg) | ![before](./images/receipt_paddleocr.png) | ![after](./images/receipt_extraction.png) |

After fine-tuning, it can output data in `JSON` format and extract corresponding information based on different `prompt`s.

> Here's a simple video of the SFT and inference process in AI Studio [PaddleOCR-VL SFT with Prompt](https://www.bilibili.com/video/BV1cTq5BvEb2/?vd_source=52a02e4f0aa6b27776bd86a6d103f2d1)

Regarding the fine-tuning of PaddleOCR-VL, [PaddleOCR-VL-0.9B SFT](https://github.com/PaddlePaddle/ERNIE/blob/release/v1.4/docs/paddleocr_vl_sft_zh.md) already provides a very detailed explanation. For fine-tuning `prompt`, the following parts are slightly different:

- Data preparation
- Model inference

## Data Preparation

The dataset used in this article is from [Spatial Dual-Modality Graph Reasoning for Key Information Extraction](https://arxiv.org/abs/2103.14470v1), introducing the [Wildreceipt dataset](https://download.openmmlab.com/mmocr/data/wildreceipt.tar). The dataset collects 1768 receipt images, for example:

![Receipt](./images/receipt.jpeg)

Since we will use [ERNIE](https://github.com/PaddlePaddle/ERNIE) to fine-tune PaddleOCR-VL, we need to prepare data in `JSON` format and corresponding image data. The following is the required data format:

```json
{
    "image_info": [{
        "matched_text_index": 0,
        "image_url": "./assets/table_example.jps"
    }, ],
    "text_info": [{
            "text": "OCR:",
            "tag": "mask"
        },
        {
            "text": "দডর মথ বধ বকসট একনজর দখই চনত পরল তর অনমন\nঠক পনতই লকয রখছ\nর নচ থকই চচয বলল কশর, "এইই; পযছ! পযছ!'\nওপর",
            "tag": "no_mask"
        },
    ]
}
```

Where,

- `image_url` is the path to the image
- `text_info` with `tag` as `mask` corresponds to the `prompt` part, which is the `TASK` type of PaddleOCR-VL
- `text_info` with `tag` as `no_mask` corresponds to the `completion` part, which is the model's output

In the original model, there are only these four types of `prompt`:

```json
{
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
}
```

But we want the model to extract corresponding information based on our instructions, so we need to customize `prompt`:

```json
{
    "image_info": [{
        "matched_text_index": 0,
        "image_url": "path/to/image.jpg"
    }],
    "text_info": [{
            "text": "OCR:{\"RECEIPT NUMBER\": \"\"}",
            "tag": "mask"
        },
        {
            "text": "{\"RECEIPT NUMBER\": \"123456789\"}",
            "tag": "no_mask"
        }
    ]
}
```

Here, the `text` with `tag` as `mask` is not `OCR:` but `OCR:{"RECEIPT NUMBER": ""}`, meaning we want the model to extract and only extract the information of the `RECEIPT NUMBER` field and output the corresponding `JSON` format result. Keeping the original `OCR:` part is to ensure that the model can recognize the `OCR:` part and only fine-tune the `{"RECEIPT NUMBER": ""}` part.

The `text` part with `tag` as `no_mask` directly outputs data in `JSON` format and corresponds to the `prompt`.

For different information formats, we design the `prompt` as follows:

```text
# Extract all information
OCR:{}

# Extract specific value as string, such as `{"NUMBER":"123456"}`
OCR:{"FIELD":""}

# Extract specific value as dictionary, such as `{"ITEM":{"NAME":"foo"}}`
OCR:{"FIELD":{}}

# Extract specific value as list, such as `{"ITMES":[{"NAME":"foo"},{"NAME":"bar"}]}`
OCR:{"FIELD":[]}
```

Here we use a VLM model (ERNIE 4.5 VL) to generate complete `JSON` format data for each image, and then randomly select some fields to form training data. For specific details on how to build the dataset, please refer to the appendix section later.

## Model Fine-tuning

The fine-tuning process is similar to [PaddleOCR-VL-0.9B SFT](https://github.com/PaddlePaddle/ERNIE/blob/release/v1.4/docs/paddleocr_vl_sft_zh.md). First, install ERNIE:

```bash
cd paddleocr_vl
git clone https://github.com/PaddlePaddle/ERNIE -b release/v1.4
cd ERNIE
python -m pip install -r requirements/gpu/requirements.txt
python -m pip install -e .
python -m pip install tensorboard
python -m pip install opencv-python-headless
python -m pip install numpy==1.26.4
```

Then, modify the configuration file and copy it to overwrite the original configuration file:

```bash
cp paddleocr_vl/sft_config/run_ocr_vl_sft_16k.yaml \
  paddleocr_vl/ERNIE/examples/configs/PaddleOCR-VL/sft/run_ocr_vl_sft_16k.yaml
```

Download the PaddleOCR-VL model, here using modelscope's SDK:

```bash
pip install modelscope
```

```python
from modelscope import snapshot_download
model_dir = snapshot_download('PaddlePaddle/PaddleOCR-VL', local_dir='paddleocr_vl/paddleocr_vl_model')
```

Finally, execute the fine-tuning command. Fine-tuning in AI Studio's A800 environment takes less than 3 hours.

> V100 environment cannot perform fine-tuning, but can perform model inference

```bash
cd paddleocr_vl/ERNIE; CUDA_VISIBLE_DEVICES=0 \
 erniekit train examples/configs/PaddleOCR-VL/sft/run_ocr_vl_sft_16k.yaml
```

Here is the training log:

![log](./images/log.png)

As you can see, the `loss` is steadily decreasing, indicating that the fine-tuning should be effective.

## Model Inference

After fine-tuning is completed, you can use the fine-tuned model for inference and:

1. Output complete information in `JSON` format
2. Output corresponding `JSON` format information based on different input fields

This provides a flexible interface for information extraction tasks.

Here are three ways to perform model inference:

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [transformers](https://github.com/huggingface/transformers)
- [PaddleOCR-VL-REC](https://github.com/megemini/PaddleOCR-VL-REC)

### Using PaddleOCR for Inference

Follow [PaddleOCR-VL-0.9B SFT](https://github.com/PaddlePaddle/ERNIE/blob/release/v1.4/docs/paddleocr_vl_sft_zh.md) for inference. First, you need to install the necessary environment:

```bash
python -m pip install -U "paddleocr[doc-parser]"
python -m pip install https://paddle-whl.bj.bcebos.com/nightly/cu126/safetensors/safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl
python -m pip install --force-reinstall opencv-python-headless
python -m pip install numpy==1.26.4
```

At this point, you still cannot directly perform model inference because PaddleOCR, which depends on PaddleX, currently only supports these four types of `prompt_label` for PaddleOCR-VL: `['ocr', 'formula', 'table', 'chart']`, and our `prompt` obviously cannot pass the code validation:

Refer to the `paddlex/inference/pipelines/paddleocr_vl/pipeline.py` file

``` python
assert prompt_label.lower() in [
    "ocr",
    "formula",
    "table",
    "chart",
], f"Layout detection is disabled (use_layout_detection=False). 'prompt_label' must be one of ['ocr', 'formula', 'table', 'chart'], but got '{prompt_label}'."

```

Here is a [patch script](https://github.com/megemini/PaddleOCR-VL-REC/blob/master/docs/paddleocr_vl/patch/patch_assert_to_warning.py) that can bypass the above restriction:

```bash
python paddleocr_vl/patch/patch_assert_to_warning.py
```

Then, copy the following files to the PaddleOCR-VL-SFT directory, and you can happily perform inference verification.

```bash
cp paddleocr_vl/paddleocr_vl_model/chat_template.jinja paddleocr_vl/PaddleOCR-VL-SFT
cp paddleocr_vl/paddleocr_vl_model/inference.yml paddleocr_vl/PaddleOCR-VL-SFT
```

Here, use the receipt mentioned above to verify the model.

```bash
python -m paddleocr doc_parser -i /home/aistudio/paddleocr_vl/data/Image_1/0/1640.jpeg \
    --vl_rec_model_name "PaddleOCR-VL-0.9B" \
    --vl_rec_model_dir "paddleocr_vl/PaddleOCR-VL-SFT" \
    --save_path="paddleocr_vl/PaddleOCR-VL-SFT_response" \
    --use_layout_detection=False \
    --prompt_label="OCR:{}"
```

Output complete information:

```json
{
    "RESTAURANT INFORMATION": {
        "NAME": "Berghotel Grosse Scheidegg",
        "ADDRESS": "3818 Grindelwald Familie R. Müller",
        "PHONE": ""
    },
    "BILL INFORMATION": {
        "BILL NUMBER": "4572",
        "DATE": "30.07.2007",
        "TIME": "13:29:17",
        "TABLE NUMBER": "7/01"
    },
    "CONSUMPTION ITEMS": {
        "2XLATTE MACCHIATO": {
            "QUANTITY": "1",
            "UNIT PRICE": "4.50",
            "TOTAL PRICE": "9.00"
        },
        "1XGLOKI": {
            "QUANTITY": "1",
            "UNIT PRICE": "5.00",
            "TOTAL PRICE": "5.00"
        },
        "1XSCHWEINSCHNITZEL": {
            "QUANTITY": "1",
            "UNIT PRICE": "22.00",
            "TOTAL PRICE": "22.00"
        },
        "1XCHÄSSPÄTZLI": {
            "QUANTITY": "1",
            "UNIT PRICE": "18.50",
            "TOTAL PRICE": "18.50"
        }
    },
    "TOTAL": "54.50",
    "CONSUMPTION TAX": "7.6% MwSt 54.50 CHF: 3.85",
    "TAX RATE": "7.6% EUR",
    "TAX AMOUNT": "36.33 EUR",
    "SERVICE STAFF": "Ursula",
    "CONTACT PHONE": "033 853 67 16",
    "EMAIL": "grossescheidegg@bluewin.ch"
}
```

Note two points:

- `use_layout_detection=False`, do not use the layout model, but directly send the image to `PaddleOCR-VL-0.9B`
- `prompt_label="OCR:{}"`, here we use our fine-tuned `prompt`, hoping the model outputs complete json format information

Then, test extracting only partial information:

```bash
python -m paddleocr doc_parser -i /home/aistudio/paddleocr_vl/data/Image_1/0/1640.jpeg \
    --vl_rec_model_name "PaddleOCR-VL-0.9B" \
    --vl_rec_model_dir "paddleocr_vl/PaddleOCR-VL-SFT" \
    --save_path="paddleocr_vl/PaddleOCR-VL-SFT_response" \
    --use_layout_detection=False \
    --prompt_label="OCR:{\"NAME\":\"\", \"ITEMS\":[]}"
```

Output:

```json
{
    "NAME": "Berghotel Grosse Scheidegg",
    "ITEMS": [{
        "NAME": "2XLATTE MACCHIATO",
        "QUANTITY": "1",
        "UNIT PRICE": "4.50",
        "TOTAL PRICE": "9.00"
    }, {
        "NAME": "1XGLOKI",
        "QUANTITY": "1",
        "UNIT PRICE": "5.00",
        "TOTAL PRICE": "5.00"
    }, {
        "NAME": "1XSCHWEINSCHNITZEL",
        "QUANTITY": "1",
        "UNIT PRICE": "22.00",
        "TOTAL PRICE": "22.00"
    }, {
        "NAME": "1XCHÄSSPÄTZLI",
        "QUANTITY": "1",
        "UNIT PRICE": "18.50",
        "TOTAL PRICE": "18.50"
    }],
    "TOTAL": "54.50",
    "CHF": "3.85",
    "INCL.": "7.6% MwSt",
    "CHF": "54.50",
    "EUR": "36.33",
    "ES BEDIENTE SIE": "Ursula",
    "MWST NR.": "430 234",
    "TEL.": "033 853 67 16",
    "FAX.": "033 853 67 19",
    "E-MAIL": "grossescheidegg@bluewin.ch"
}]
```

As you can see, the model can basically follow our instructions to extract corresponding information.

However, there are still some flaws, such as:

- The model not only extracted `NAME` and `ITEMS` information but also extracted several other field information, indicating that the fine-tuned model still has some instruction-following flaws, which can be solved by expanding the dataset to strengthen training.
- The `JSON` format output by the model is incomplete, which is also a common problem with large models. Some tools can alleviate such problems, such as [json_repair](https://github.com/mangiucugna/json_repair/)

### Using transformers for Inference

You can use the transformers library for information extraction, refer to [[Model] Add PaddleOCR-VL Model Support by zhang-prog](https://github.com/huggingface/transformers/pull/42178)

> Note, the model directory generated after fine-tuning has not been synchronized and updated yet. When using the transformers library for information extraction, you need to first download the latest model from [huggingface](https://huggingface.co/PaddlePaddle/PaddleOCR-VL/tree/main), then rename the fine-tuned model file `model-00001-of-00001.safetensors` to `model.safetensors`, and put it (and overwrite) in the downloaded model directory.

```python
from transformers import pipeline

pipe = pipeline(
    "image-text-to-text", 
    model="./PaddleOCR_VL_SFT/PaddleOCR-VL", # downloaded model directory
    dtype="bfloat16")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "path/to/image.jpg"},
            {"type": "text", "text": "OCR:{}"},
        ]
    }
]
result = pipe(text=messages)
print(result)

```

If there is insufficient GPU memory, you can try the following quantization method:

```python
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
import torch

path = "./PaddleOCR_VL_SFT/PaddleOCR-VL", # downloaded model directory
processor = AutoProcessor.from_pretrained(path, local_files_only=True, use_fast=True)

# 4-bit quantization configuration, significantly reducing memory usage
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
model = AutoModelForImageTextToText.from_pretrained(
    path,
    quantization_config=quantization_config,
    # device_map="auto",
    local_files_only=True
)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "path/to/image.jpg"},
            {"type": "text", "text": "OCR:{\"NAME\": \"\"}"},
        ]
    }
]
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=100)
result = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:-1])
print(result)

```

### Using PaddleOCR-VL-REC for Information Extraction

You can use [PaddleOCR-VL-REC](https://github.com/megemini/PaddleOCR-VL-REC) for information extraction:

```python
from paddleocr_vl_rec import PaddleOCRVLRec

# Initialize the recognizer
recognizer = PaddleOCRVLRec(
    model_dir="path/to/your/model"
)

# Use dict as query (will be converted to JSON string)
# Return JSON format (using json_repair to parse the result)
result_json = recognizer.predict(
    image="/path/to/your/image.jpg",
    query={"NAME":"", "ITEMS":[]},
    return_json=True
)
# result_json is a dictionary object
print(type(result_json))  # <class 'dict'>
print(result_json)

# Use list as query (will be converted to {"ITEM1":"", "ITEM2":""} format)
result_json = recognizer.predict(
    image="/path/to/your/image.jpg",
    query=["ITEM1", "ITEM2"],
    return_json=True
)
print(result_json)

recognizer.close()

```

The tool simplifies the calling process of the PaddleOCR-VL model in PaddleOCR, skips the preprocessing of the PP-DocLayoutV2 model, directly uses the PaddleOCR-VL-0.9B model for inference, and uses [json_repair](https://github.com/mangiucugna/json_repair/) to repair the results, making it easier to use.

For example, using [PaddleOCR-VL-REC](https://github.com/megemini/PaddleOCR-VL-REC) to extract partial information from the above receipt, you get a complete `JSON` result:

![full](./images/receipt_extraction_full.png)

![after](./images/receipt_extraction.png)

## Summary

This article introduces how to implement information extraction tasks by fine-tuning the prompts of PaddleOCR-VL. The main methods include:

1. **Prompt Design**: By designing prompt templates, the model can flexibly output `JSON` format information for different fields.
2. **Model Fine-tuning**: Utilizing the fine-tuning capability of PaddleOCR-VL to make it learn to generate corresponding outputs based on different prompts.

Compared with traditional information extraction methods (such as NER + relation extraction), this method has better integration and flexibility.

## Appendix

### 1. Dataset

There are many application scenarios for information extraction. Here, we take the [VAT Ordinary Invoice](https://aistudio.baidu.com/datasetdetail/125158) data as an example.

> You can refer to the article [Invoice Key Information Extraction Based on VI-LayoutXLM](https://bbs.huaweicloud.com/blogs/383854), which provides a relatively complete explanation of fine-tuning the PaddleOCR model for information extraction.

However, the dataset's annotation for `Relation Extraction` is still relatively simple. For example:

![VAT Ordinary Invoice](images/re.jpg)

Here only `名称` (Name) is annotated, without specifying whether it is `购买方名称` (Buyer Name) or `销售方名称` (Seller Name).

As mentioned earlier, we can use PaddleOCR-VL as a VLM model, so we can let a more capable VLM model `teach` PaddleOCR-VL to recognize `购买方名称` (Buyer Name) and `销售方名称` (Seller Name).

Data can be generated through the `ernie-4.5-turbo-vl-preview` model, refer to the script [extract_ner.py](https://github.com/megemini/PaddleOCR-VL-REC/blob/master/docs/paddleocr_vl/tools/extract_ner/extract_ner.py).

``` python

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multimodal image recognition script
Identify image information by calling OpenAI API and return JSON format data
Supports local images and multimodal large model processing
"""
...

class MultimodalImageRecognizer:
    """Multimodal Image Recognizer"""
    ...

    def recognize_image(
        self,
        image_input: Union[str, bytes],
        prompt: str,
        system_prompt: str,
        max_tokens: int = 2048
    ) -> Dict[str, Any]:
        """
        Identify image information

        Args:
            image_input: Image path, URL or base64 encoding
            prompt: User prompt
            system_prompt: System prompt
            max_tokens: Maximum number of tokens

        Returns:
            JSON format data of recognition results
        """
        try:
            # Create multimodal message
            content = self.create_multimodal_message(prompt, image_input)

            # Build message list
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ]

            logger.info(f"Start calling API to recognize image, model: {self.model}")

            # Call API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.2
            )

    ...

    def analyze_image(
        self,
        image_input: Union[str, bytes],
        analysis_type: str = "document"
    ) -> Dict[str, Any]:
        """
        Analyze image content (simplified version)

        Args:
            image_input: Image path, URL or base64 encoding
            analysis_type: Analysis type, fixed as "document"

        Returns:
            JSON format data of analysis results
        """
        # Uniformly use document analysis prompt
        prompt = "Please analyze all information in this document image and return complete JSON format data. If a field has no value, keep this field with an empty value. Note: All values should be returned as strings, do not use numeric types, etc."
        system_prompt = '''
You are a professional document analysis assistant capable of accurately analyzing document content and returning structured JSON data.

Note: The language of the data should be consistent with the language of the document.
Note: Need to maintain the complete field hierarchy, do not put all fields in the first-level fields.
Note: Do not include comments in the JSON data, nor any explanations or descriptions.
Note: Special characters need to be escaped.

Note: For option fields, only keep the selected field value. If not selected, set it to empty.
For example, `业务类型` (Business Type) includes options such as `账户开户、账户登记` (Account Opening, Account Registration). If `账户登记` is selected in the document, return `{"业务类型"："账户登记"}`, do not return other options like `账户开户`.
Another example, `业务类型` includes options such as `账户开户、账户登记`. If no option is marked as selected in the document, return `{"业务类型"：""}`, that is, only keep the key, no need for a value.
...
'''

        return self.recognize_image(
            image_input=image_input,
            prompt=prompt,
            system_prompt=system_prompt
        )

...
```

Use the [batch_extract_ner.py](https://github.com/megemini/PaddleOCR-VL-REC/blob/master/docs/paddleocr_vl/tools/extract_ner/batch_extract_ner.py) script to batch generate data. The final generated data is as follows:

``` json

{
  "image": "/media/shun/bigdata/Dataset/增值税普通发票/zzsptfp/b0.jpg",
  "data": {
    "发票名称": "广东增值税专用发票",
    "发票编号": "12271524",
    "发票代码": "4400154130",
    "开票日期": "2016年06月12日",
    "购买方": {
      "名称": "深圳市购机汇网络有限公司",
      "纳税人识别号": "440300083885931",
      "地址、电话": "深圳市龙华新区民治街道民治大道展滔科技大厦A12070755-23806606",
      "开户行及账号": "中国工商银行股份有限公司深圳园岭支行4000024709200172809"
    },
    "密码区": "<<1<//3*26-++936-9<9*575>39 -<5//81>84974<00+7>2*0*53-+ +125*++9+-///5-7+/-0>8<9815 5<3/8*+//81/84+>6>4*36>4538",
    "货物或应税劳务、服务名称": [
      {
        "名称": "小米 红米3 全网通版 时尚金色",
        "规格型号": "红米3",
        "单位": "个",
        "数量": "5",
        "单价": "597.43589744",
        "金额": "2987.18",
        "税率": "17%",
        "税额": "507.82"
      },
      {
        "名称": "移动联通电信4G手机 双卡双待",
        "规格型号": "",
        "单位": "",
        "数量": "",
        "单价": "",
        "金额": "",
        "税率": "",
        "税额": ""
      }
    ],
    "合计": {
      "金额": "￥2987.18",
      "税额": "￥507.82"
    },
    "价税合计（大写）": "叁仟肆佰玖拾伍圆整",
    "价税合计（小写）": "￥3495.00",
    "销售方": {
      "名称": "广州晶东贸易有限公司",
      "纳税人识别号": "91440101664041243T",
      "地址、电话": "广州市黄埔区九龙镇九龙工业园凤凰三横路99号 66215500",
      "开户行及账号": "工行北京路支行3602000919200384952"
    },
    "备注": "dd42982413947(00001,1952)7996有限",
    "收款人": "王梅",
    "复核": "张雪",
    "开票人": "陈秋燕",
    "销售方（章）": "广州晶东贸易有限公司 发票专用章"
  }
}

```

The generated data information here is much richer than the original annotation information. Although there are some flaws (for example, `货物或应税劳务、服务名称` should only have one record), it does not hinder the progress of fine-tuning experiments.

> The processed data has been uploaded to [VAT Ordinary Invoice and JSON Format Information](https://aistudio.baidu.com/dataset/detail/363136/intro).

### 2. Prompts

You can use [process_ner_dataset.py](https://github.com/megemini/PaddleOCR-VL-REC/blob/master/docs/paddleocr_vl/tools/process_ner_dataset.py) to generate complete training data, including randomly generated prompts:

```bash
python paddleocr_vl/tools/process_ner_dataset.py paddleocr_vl/data/zzsptfp \
  -o paddleocr_vl/output.jsonl \
  -n 10 \
  -p /media/shun/bigdata/Dataset/增值税普通发票 \
  -u /home/aistudio/paddleocr_vl/data/zzsptfp
```

Then, use [split_jsonl.py](https://github.com/megemini/PaddleOCR-VL-REC/blob/master/docs/paddleocr_vl/tools/split_jsonl.py) to split the training dataset and validation dataset:

```bash
python paddleocr_vl/tools/split_jsonl.py paddleocr_vl/output.jsonl \
  paddleocr_vl/output \
  --train_ratio 0.9 \
  --seed 123
```

The final generated data is as follows:

```json
{
    "image_info": [
        {
            "matched_text_index": 0,
            "image_url": "/home/aistudio/paddleocr_vl/data/zzsptfp/zzsptfp/b175.jpg"
        }
    ],
    "text_info": [
        {
            "text": "OCR:{\"发票名称\": \"\"}",
            "tag": "mask"
        },
        {
            "text": "{\"发票名称\": \"广东增值税专用发票\"}",
            "tag": "no_mask"
        }
    ]
}
```

The differences between the generated training data and [PaddleOCR-VL-0.9B SFT](https://github.com/PaddlePaddle/ERNIE/blob/release/v1.4/docs/paddleocr_vl_sft_zh.md) are:

- The `text` of `mask` is not just `OCR:`, but also includes the field information to be extracted later
- The `text` of `no_mask` is complete `JSON` format information, not a plain text

### 3. Configuration File Example

```yaml
### data
train_dataset_type: "erniekit"
eval_dataset_type: "erniekit"
train_dataset_path: "/home/aistudio/paddleocr_vl/output_train.jsonl"
train_dataset_prob: "1.0"
eval_dataset_path: "/home/aistudio/paddleocr_vl/output_val.jsonl"
eval_dataset_prob: "1.0"
max_seq_len: 16384
num_samples_each_epoch: 6000000
use_pic_id: False
sft_replace_ids: True
sft_image_normalize: True
sft_image_rescale: True
image_dtype: "float32"

### model
model_name_or_path: "/home/aistudio/paddleocr_vl/paddleocr_vl_model"
fine_tuning: Full
multimodal: True
use_flash_attention: True
use_sparse_flash_attn: True

### finetuning
# base
stage: OCR-VL-SFT
seed: 23
do_train: True
# do_eval: True
distributed_dataloader: False
dataloader_num_workers: 8
prefetch_factor: 10
batch_size: 1
packing_size: 8
packing: True
padding: False
num_train_epochs: 2
max_steps: 80
# eval_batch_size: 1
# eval_iters: 50
# eval_steps: 100
# evaluation_strategy: steps
save_steps: 20
save_total_limit: 5
save_strategy: steps
logging_steps: 1
release_grads: True
gradient_accumulation_steps: 8
logging_dir: /home/aistudio/paddleocr_vl/PaddleOCR-VL-SFT/tensorboard_logs/
output_dir: /home/aistudio/paddleocr_vl/PaddleOCR-VL-SFT
disable_tqdm: True

# train
warmup_steps: 1
learning_rate: 5.0e-6
lr_scheduler_type: cosine
min_lr: 5.0e-7
layerwise_lr_decay_bound: 1.0
from_scratch: 0

# optimizer
weight_decay: 0.1
adam_epsilon: 1.0e-8
adam_beta1: 0.9
adam_beta2: 0.95

# performance
tensor_parallel_degree: 1
pipeline_parallel_degree: 1
sharding_parallel_degree: 1
sharding: stage1
sequence_parallel: False
pipeline_parallel_config: enable_delay_scale_loss enable_release_grads disable_partial_send_recv
recompute: True
recompute_granularity: "full"
recompute_use_reentrant: True
compute_type: bf16
fp16_opt_level: O2
disable_ckpt_quant: True
# amp_master_grad: True
amp_custom_white_list:
  - lookup_table
  - lookup_table_v2
  - flash_attn
  - matmul
  - matmul_v2
  - fused_gemm_epilogue
amp_custom_black_list:
  - reduce_sum
  - softmax_with_cross_entropy
  - c_softmax_with_cross_entropy
  - elementwise_div
  - sin
  - cos
unified_checkpoint: True
# unified_checkpoint_config: async_save
convert_from_hf: True
save_to_hf: True
```

### 4. Resource Links

- AI Studio Project Address: [Fine-tuning PaddleOCR-VL with New Approach -- Prompt and Information Extraction](https://aistudio.baidu.com/projectdetail/9857242), which can be run directly in AI Studio's A100 environment (V100 environment can only perform model inference, cannot perform fine-tuning)
- Model Address: [megemini/PaddleOCR-VL-Receipt](https://modelscope.cn/models/megemini/PaddleOCR-VL-Receipt/summary)
- Inference Tool: [PaddleOCR-VL-REC](https://github.com/megemini/PaddleOCR-VL-REC)
- Dataset: [Wildreceipt arxiv](https://arxiv.org/abs/2103.14470v1), [Wildreceipt dataset](https://download.openmmlab.com/mmocr/data/wildreceipt.tar)
- SFT Process Video: [PaddleOCR-VL SFT with Prompt](https://www.bilibili.com/video/BV1cTq5BvEb2/?vd_source=52a02e4f0aa6b27776bd86a6d103f2d1)