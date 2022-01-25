from typing import Optional
from pydantic import BaseModel, Field


class TextGenerationPredictPayload(BaseModel):
    text_inputs: str
    max_length: Optional[int] = Field(None, title="The maximum length of the sequence to be generated.")
    min_length: Optional[int] = Field(None, title="The minimum length of the sequence to be generated.")
    do_sample: Optional[bool] = Field(None, title="Whether or not to use sampling ; use greedy decoding otherwise.")
    early_stopping: Optional[bool] = Field(None,
                                           title="Whether to stop the beam search when at least num_beams sentences are finished per batch or not.")
    num_beams: Optional[int] = Field(None, title="Number of beams for beam search. 1 means no beam search.")
    temperature: Optional[float] = Field(None, title="The value used to module the next token probabilities.")
    top_k: Optional[int] = Field(None,
                                 title="The number of highest probability vocabulary tokens to keep for top-k-filtering.")
    top_p: Optional[float] = Field(None,
                                   title="If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.")
    repetition_penalty: Optional[float] = Field(None,
                                                title="The parameter for repetition penalty. 1.0 means no penalty. See this paper for more details.")
    length_penalty: Optional[float] = Field(None,
                                            title="xponential penalty to the length. 1.0 means no penalty. Set to values < 1.0 in order to encourage the model to generate shorter sequences, to a value > 1.0 in order to encourage the model to produce longer sequences.")
    no_repeat_ngram_size: Optional[int] = Field(None,
                                                title="If set to int > 0, all ngrams of that size can only occur once.")
    num_return_sequences: Optional[int] = Field(None,
                                                title="The number of independently computed returned sequences for each element in the batch.")
