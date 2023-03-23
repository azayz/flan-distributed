from random import random
from typing import Optional

import numpy as np
import torch
from docarray import DocumentArray, BaseDocument
from jina import Executor, requests, Flow
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


class InputSchema(BaseDocument):
    text: Optional[str]


class OutputSchema(BaseDocument):
    text: Optional[str]


class LLM(Executor):

    def __init__(self, model_name: str, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to('cuda:0')

    @requests
    def generate(self, docs: DocumentArray[InputSchema], **kwargs) -> DocumentArray[OutputSchema]:
        outputs = DocumentArray[OutputSchema]()
        max_length = 100
        num_return_sequences = 1
        dict_input = dict(docs[0])
        input_ids = self.tokenizer.encode(dict_input['text'], return_tensors="pt")

        # Move input to the first GPU
        input_ids = input_ids.to("cuda:0")

        # Generate output with the model
        result = self.model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
        )

        # Move generated output to the CPU
        result = result.cpu()

        # Decode the outputs to obtain text
        generated_text = [
            self.tokenizer.decode(res, skip_special_tokens=True) for res in result
        ]

        model_output = OutputSchema(text=' '.join(generated_text))
        outputs.extend([model_output])
        return outputs


f = Flow().add(
    uses=LLM,
    uses_with={
        'model_name': "google/flan-t5-xl",
    },
)

with f:
    docs = f.post(
        on='/',
        inputs=DocumentArray([InputSchema(
            text="Translate the following English text to French: 'Hey, how are you?'"
        )]),
        return_type=DocumentArray[OutputSchema],
    )
    outputs = dict(docs[0])
    print(outputs)
    docs.summary()
