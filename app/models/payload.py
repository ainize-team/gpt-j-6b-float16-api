from typing import Optional
from pydantic import BaseModel, Field


class TextGenerationPredictPayload(BaseModel):
    text_inputs: str = Field(..., title="Input text to generate sentences.", example="My name is Julien and I like to")
    max_length: Optional[int] = Field(None, title="The maximum length of the sequence to be generated.", example=20)
    min_length: Optional[int] = Field(None, title="The minimum length of the sequence to be generated.", example=10)
    do_sample: Optional[bool] = Field(None, title="Whether or not to use sampling ; use greedy decoding otherwise.",
                                      example=False)
    early_stopping: Optional[bool] = Field(None,
                                           title="Whether to stop the beam search when at least num_beams sentences are finished per batch or not.",
                                           example=False)
    num_beams: Optional[int] = Field(None, title="Number of beams for beam search. 1 means no beam search.", example=1)
    temperature: Optional[float] = Field(None, title="The value used to module the next token probabilities.",
                                         example=1.0)
    top_k: Optional[int] = Field(None,
                                 title="The number of highest probability vocabulary tokens to keep for top-k-filtering.",
                                 example=50)
    top_p: Optional[float] = Field(None,
                                   title="If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.",
                                   example=1.0)
    repetition_penalty: Optional[float] = Field(None,
                                                title="The parameter for repetition penalty. 1.0 means no penalty. See this paper for more details.",
                                                example=1.0)
    length_penalty: Optional[float] = Field(None,
                                            title="xponential penalty to the length. 1.0 means no penalty. Set to values < 1.0 in order to encourage the model to generate shorter sequences, to a value > 1.0 in order to encourage the model to produce longer sequences.",
                                            example=1.0)
    no_repeat_ngram_size: Optional[int] = Field(None,
                                                title="If set to int > 0, all ngrams of that size can only occur once.",
                                                example=1.0
                                                )
    num_return_sequences: Optional[int] = Field(None,
                                                title="The number of independently computed returned sequences for each element in the batch.",
                                                example=1
                                                )
