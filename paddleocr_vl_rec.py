#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PaddleOCR VL Recognition Script
Only includes vl_rec_model initialization and inference functionality.
"""

import sys
import os
import warnings
from typing import Optional, Dict, Any, Union, Literal

import json
import json_repair
import numpy as np

# Add parent directory to path to import paddlex
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paddlex.inference import create_predictor

PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
}

class PaddleOCRVLRec:
    """PaddleOCR VL Recognition class that only handles vl_rec_model"""
    
    def __init__(
        self,
        model_name: str = "PaddleOCR-VL-0.9B",
        model_dir: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 1,
        genai_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the VL Recognition model.
        
        Args:
            model_name (str): Name of the model. Default is "PaddleOCR-VL-0.9B"
            model_dir (str, optional): Path to the model directory. Default is None.
            device (str, optional): Device to run on (e.g., 'cpu', 'gpu:0'). Default is None.
            batch_size (int): Batch size for inference. Default is 1.
            genai_config (dict, optional): GenAI configuration. Default is None.
        """
        print(f"Initializing VL Recognition model: {model_name}")
        
        self.vl_rec_model = create_predictor(
            model_name=model_name,
            model_dir=model_dir,
            device=device,
            batch_size=batch_size,
            genai_config=genai_config,
        )
        
        print("VL Recognition model initialized successfully!")
    
    def predict(
        self,
        image: Union[str, np.ndarray],
        prompt_label: Union[Literal["ocr", "table", "formula", "chart"], None] = "ocr",
        query: Union[str, list, dict, None] = None,
        skip_special_tokens: bool = True,
        max_new_tokens: Optional[int] = 4096,
        return_json: bool = False,
        **kwargs
    ) -> Union[str, Dict[str, Any]]:
        """
        Predict text from an image using the VL Recognition model.

        Args:
            image (str or np.ndarray): Path to image file or numpy array of the image
            prompt_label (str or None): Type of recognition task. Options: "ocr", "table", "formula", "chart". Default is "ocr".
                                       Provides the base prompt from predefined templates.
            query (str, list, dict, or None): Additional custom text/data to append to the prompt. Default is None.
                                 - If str: Appended directly to the base prompt.
                                 - If list: Converted to JSON format {"item1":"", "item2":""} and appended to base prompt.
                                 - If dict: JSON dumped and appended to base prompt.
            skip_special_tokens (bool): Whether to skip special tokens. Default is True.
            max_new_tokens (int, optional): Maximum number of new tokens to generate. Default is 4096.
            return_json (bool): Whether to parse and return result as JSON. Default is False.
                               If True, will use json_repair (preferred) or json to parse the result.
            **kwargs: Additional keyword arguments for the model.

        Returns:
            str or dict: Recognized text from the image (str if return_json=False, dict if return_json=True)
        """
        # Prepare input for the model
        # Build query: prompt_label provides the base prompt, query can be appended
        base_prompt = (PROMPTS.get(prompt_label, "") if prompt_label else "")
        
        # Convert query to string format
        if query is not None:
            if isinstance(query, dict):
                query_str = json.dumps(query, ensure_ascii=False)
            elif isinstance(query, list):
                # Convert list to dict format: {"item1":"", "item2":"", ...}
                query_dict = {item: "" for item in query}
                query_str = json.dumps(query_dict, ensure_ascii=False)
            else:
                query_str = str(query)
        else:
            query_str = ""
        
        final_query = base_prompt + query_str
        
        model_input = [{
            "image": image,
            "query": final_query,
        }]

        # Prepare kwargs
        inference_kwargs = {
            "use_cache": True,
            "skip_special_tokens": skip_special_tokens,
            **kwargs,
        }

        if max_new_tokens is not None:
            inference_kwargs["max_new_tokens"] = max_new_tokens

        # Run prediction
        results = list(self.vl_rec_model.predict(model_input, **inference_kwargs))

        # Extract text result
        if results and len(results) > 0:
            result_str = results[0].get("result", "")
            if result_str is None:
                result_str = ""
        else:
            result_str = ""

        # Parse as JSON if requested
        if return_json and result_str:
            return self._parse_json(result_str)

        return result_str

    def _parse_json(self, text: str) -> Union[Dict[str, Any], str]:
        """
        Parse text as JSON using json_repair (preferred) or json.

        Args:
            text (str): Text to parse as JSON

        Returns:
            dict or str: Parsed JSON object, or original text if parsing fails
        """
        try:
            return json_repair.loads(text)
        except Exception as e:
            warnings.warn(f"Failed to parse JSON: {e}. Returning original text.")
            return text
    
    def close(self):
        """Close the model and release resources"""
        if hasattr(self, 'vl_rec_model'):
            self.vl_rec_model.close()
            print("VL Recognition model closed.")


def main():
    """Example usage of PaddleOCRVLRec"""
    import argparse

    parser = argparse.ArgumentParser(description="PaddleOCR VL Recognition Script")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model_name", type=str, default="PaddleOCR-VL-0.9B",
                        help="Model name (default: PaddleOCR-VL-0.9B)")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Path to model directory (optional)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (e.g., 'cpu', 'gpu:0')")
    parser.add_argument("--prompt_label", type=str, default="ocr",
                        choices=["ocr", "table", "formula", "chart"],
                        help="Type of recognition task (default: 'ocr')")
    parser.add_argument("--query", type=str, default=None,
                        help="Custom query prompt (overrides prompt_label if provided)")
    parser.add_argument("--max_new_tokens", type=int, default=4096,
                        help="Maximum new tokens (default: 4096)")
    parser.add_argument("--return_json", action="store_true",
                        help="Parse and return result as JSON")

    args = parser.parse_args()

    # Initialize recognizer
    recognizer = PaddleOCRVLRec(
        model_name=args.model_name,
        model_dir=args.model_dir,
        device=args.device,
    )

    try:
        # Run prediction
        print(f"\nProcessing image: {args.image}")
        print(f"Prompt label: {args.prompt_label}")
        if args.query:
            print(f"Custom query: {args.query}")
        print(f"Return JSON: {args.return_json}")
        print("-" * 50)

        result = recognizer.predict(
            image=args.image,
            prompt_label=args.prompt_label,
            query=args.query,
            max_new_tokens=args.max_new_tokens,
            return_json=args.return_json,
        )

        print("Recognition Result:")
        print(result)
        print("-" * 50)

    finally:
        # Clean up
        recognizer.close()


if __name__ == "__main__":
    main()

