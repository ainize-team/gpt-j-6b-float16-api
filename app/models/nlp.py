import gc
import torch

from fastapi import HTTPException
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from loguru import logger

from models.payload import TextGenerationPredictPayload
from models.prediction import TextGenerationResult


class TextGenerationModel:
    def __init__(self, model_name_or_path: str, revision: str, is_fp16: bool, low_cpu_mem_usage: bool):
        if not revision:
            revision = None
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            revision=revision,
            low_cpu_mem_usage=low_cpu_mem_usage
        )
        if is_fp16:
            self.model = self.model.half()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.model.eval()
        self.model_max_length = 1024
        if hasattr(self.model.config, "n_positions"):
            self.model_max_length = self.model.config.n_positions
        elif hasattr(self.model.config, "max_position_embeddings"):
            self.model_max_length = self.model.config.max_position_embeddings

    def predict(self, request: TextGenerationPredictPayload) -> TextGenerationResult:
        request_dict = request.dict()
        if len(request.text_inputs) > self.model_max_length * 128:
            logger.error(f"`text_inputs` length is {len(request.text_inputs)}")
            raise HTTPException(status_code=413, detail="`text_inputs` is too long to generate")
        inputs = self.tokenizer.encode(request.text_inputs, return_tensors='pt')
        if inputs.shape[1] > self.model_max_length:
            logger.error(f"encoded sequence length is {inputs.shape[1]}")
            raise HTTPException(status_code=413, detail="`text_inputs` is too long to generate")
        try:
            request_dict["inputs"] = inputs.to(device=self.device, non_blocking=True)
            del request_dict["text_inputs"]
            gen_tokens = self.model.generate(**request_dict)
            del request_dict
            generated_text = self.tokenizer.batch_decode(gen_tokens.tolist(), skip_special_tokens=True)
            return TextGenerationResult(generated_text=generated_text)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal Server Error : {e}")
        finally:
            if self.device == "cuda":
                if "inputs" in request_dict:
                    del request_dict["inputs"]
                if 'gen_tokens' in locals():
                    del gen_tokens
                torch.cuda.empty_cache()
            del request_dict
            gc.collect()
            logger.info("clean memory")


