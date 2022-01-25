import torch

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from loguru import logger

from models.payload import TextGenerationPredictPayload
from models.prediction import TextGenerationResult


class TextGenerationModel:
    def __init__(self, model_name_or_path: str, revision: str, is_fp16: bool):
        if revision:
            self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, revision=revision)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        if is_fp16:
            self.model = self.model.half()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.model.eval()

    def predict(self, request: TextGenerationPredictPayload) -> TextGenerationResult:
        request_dict = request.dict()
        inputs = self.tokenizer.encode(request.text_inputs, return_tensors='pt').to(device=self.device, non_blocking=True)
        logger.info(f"Input Text: {request.text_inputs}")
        request_dict["inputs"] = inputs
        del request_dict["text_inputs"]
        gen_tokens = self.model.generate(**request_dict)
        generated_text = self.tokenizer.batch_decode(gen_tokens.tolist(), skip_special_tokens=True)
        return TextGenerationResult(generated_text=generated_text)
