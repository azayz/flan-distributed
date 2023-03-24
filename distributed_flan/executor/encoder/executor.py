import io

import torch
from transformers import AutoModelForSeq2SeqLM
from typing import Optional

from jina import Executor, requests, DocumentArray
from docarray.typing import TorchTensor
from docarray import BaseDocument


class InputSchema(BaseDocument):
    input_ids: Optional[TorchTensor]
    attention_mask: Optional[TorchTensor]
    encoder_hidden_states: Optional[TorchTensor]
    encoder_attention_mask: Optional[TorchTensor]
    inputs_embeds: Optional[TorchTensor]
    head_mask: Optional[TorchTensor]
    cross_attn_head_mask: Optional[TorchTensor]
    past_key_values: Optional[bytes]
    use_cache: Optional[bool]
    output_attentions: Optional[bool]
    output_hidden_states: Optional[bool]
    return_dict: Optional[bool]


class OutputSchema(BaseDocument):
    last_hidden_state: Optional[TorchTensor]
    past_key_values: Optional[bytes] = None
    hidden_states: Optional[TorchTensor] = None
    attentions: Optional[TorchTensor] = None
    cross_attentions: Optional[TorchTensor] = None


class EncoderExecutor(Executor):
    def __init__(self, model_name: str, device_map: dict, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.device_map = {}
        for k, v in device_map.items():
            self.device_map[int(k)] = v
        self.model_encoder = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name
        ).encoder
        self.model_encoder.parallelize(self.device_map)
        print('model sent to GPU')

    @requests
    def encode(
        self, docs: DocumentArray[InputSchema], **kwargs
    ) -> DocumentArray[OutputSchema]:
        outputs = DocumentArray[OutputSchema]()

        for doc in docs:
            doc_as_dict = dict(doc)
            doc_as_dict.pop('id')
            for k, v in doc_as_dict.items():
                if isinstance(v, TorchTensor):
                    doc_as_dict[k] = v.to('cuda:0')
            model_output = self.model_encoder(**doc_as_dict)

            if "past_key_values" in model_output:
                buffer = io.BytesIO()
                torch.save(model_output["past_key_values"], buffer)
                serialized_tensor = buffer.getvalue()
                model_output["past_key_values"] = serialized_tensor
            model_output = OutputSchema(**model_output)
            outputs.extend([model_output])
        return outputs
