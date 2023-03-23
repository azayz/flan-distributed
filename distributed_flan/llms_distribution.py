import asyncio
import io

from typing import Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
import os
import random
import numpy as np
from transformers.models.t5.modeling_t5 import T5Stack
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from docarray.typing import TorchTensor
from docarray import BaseDocument
from jina import Executor, requests, DocumentArray, Deployment, Flow, Client
from jina.serve.runtimes.gateway.http.fastapi import FastAPIBaseGateway


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


def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


class Encoder(Executor):

    def __init__(self, model_name: str, device_map: dict, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.device_map = {}
        for k, v in device_map.items():
            self.device_map[int(k)] = v
        self.model_encoder = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).encoder
        print('model loaded to cpu')
        self.model_encoder.parallelize(self.device_map)
        print('model sent to GPU')

    @requests
    def encode(self, docs: DocumentArray[InputSchema], **kwargs) -> DocumentArray[OutputSchema]:
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


class T5StackWrapper(T5Stack):
    def __init__(self, config, executor_client, embed_tokens=None):
        super().__init__(config=config, embed_tokens=embed_tokens)
        self.executor_client = executor_client

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            inputs_embeds=None,
            head_mask=None,
            cross_attn_head_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):

        if past_key_values:
            buffer = io.BytesIO()
            torch.save(past_key_values, buffer)
            past_key_values = buffer.getvalue()
        # format f doc array v2 somehow
        docs = self.executor_client.post(
            on='/encode',
            inputs=DocumentArray([InputSchema(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )]),
            parameters={},
            return_type=DocumentArray[OutputSchema],
        )
        outputs = dict(docs[0])
        if outputs["past_key_values"] and len(outputs["past_key_values"]) > 0:
            buffer = io.BytesIO(outputs["past_key_values"])
            outputs["past_key_values"] = torch.load(buffer)
        outputs.pop('id')
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                outputs[key] = value.to('cuda:0')
        return BaseModelOutputWithPastAndCrossAttentions(
            **outputs
        )


'''
set_random_seeds()

with Deployment(uses=Encoder, uses_with={
    'model_name': "google/flan-t5-xl",
    'device_map': {
        0: list(range(0, 8)),
        1: list(range(8, 16)),
        2: list(range(16, 24)),
    }}, port=12346) as dep:
    config, _ = AutoConfig.from_pretrained(
        "google/flan-t5-xl",
        return_unused_kwargs=True,
        trust_remote_code=True,
    )
    t5stack_wrapper = T5StackWrapper(config, dep, embed_tokens=torch.nn.Embedding(config.vocab_size, config.d_model,
                                                                                  device='cuda:0'))
    output = t5stack_wrapper(
        input_ids=torch.tensor(
            [[30355, 15, 8, 826, 1566, 1499, 12, 2379, 10, 3, 31, 3845, 63, 6, 149, 33, 25, 58, 31, 1]],
            device='cuda:0', dtype=torch.int64),
        attention_mask=torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], device='cuda:0',
                                    dtype=torch.int64),
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True
    )
    print(output)'''


class Decoder(Executor):
    def __init__(self, model_name: str, device_map: dict, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.device_map = {int(k): v for k, v in device_map.items()}
        self.model_decoder = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).decoder
        print('model loaded to cpu')
        self.model_decoder.parallelize(self.device_map)
        print('model sent to GPU')

    @requests
    def decode(self, docs: DocumentArray[InputSchema], **kwargs) -> DocumentArray[OutputSchema]:
        outputs = DocumentArray[OutputSchema]()
        for doc in docs:
            inputs = dict(doc)
            inputs.pop('id')
            if inputs["past_key_values"] and len(inputs["past_key_values"]) > 0:
                buffer = io.BytesIO(inputs["past_key_values"])
                inputs["past_key_values"] = torch.load(buffer)
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to("cuda:0")
            model_output = self.model_decoder(**inputs)
            buffer = io.BytesIO()

            torch.save(model_output["past_key_values"], buffer)
            serialized_tensor = buffer.getvalue()
            model_output["past_key_values"] = serialized_tensor
            output = OutputSchema(**model_output)
            outputs.extend([output])
        return outputs


'''
with Deployment(uses=Decoder, uses_with={
    'model_name': "google/flan-t5-xl",
    'device_map': {
        0: list(range(0, 8)),
        1: list(range(8, 16)),
        2: list(range(16, 24)),
    }}, port=12346) as dep:
    config, _ = AutoConfig.from_pretrained(
        "google/flan-t5-xl",
        return_unused_kwargs=True,
        trust_remote_code=True,
    )
    t5stack_wrapper = T5StackWrapper(config, dep,
                                     embed_tokens=torch.nn.Embedding(config.vocab_size, config.d_model).to('cuda:0'))
    output = t5stack_wrapper(
        input_ids=torch.tensor([[6]], device='cuda:0'),
        attention_mask=None,
        encoder_hidden_states=torch.rand((1, 20, 2048), device='cuda:0') - 0.5,
        encoder_attention_mask=torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
                                            device='cuda:0', dtype=torch.int64),
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=True,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True
    )
    print(output)
'''

from fastapi import FastAPI
from uvicorn import Server, Config
from jina import DocumentArray, Gateway


class MyGateway(Gateway):
    def __init__(self, model_name, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        config, _ = AutoConfig.from_pretrained(
            model_name,
            return_unused_kwargs=True,
            trust_remote_code=True,
        )
        embedding_layer = torch.nn.Embedding(config.vocab_size, config.d_model).to(
            'cuda:0')
        encoder_client = Client(port=12346)
        decoder_client = Client(port=12347)
        self.model.encoder = T5StackWrapper(config, encoder_client,
                                            embed_tokens=embedding_layer)

        self.model.decoder = T5StackWrapper(config, decoder_client,
                                            embed_tokens=embedding_layer)
        self.model.lm_head = self.model.lm_head.to('cuda:0')
        self.model.parallel = True

    async def setup_server(self):
        # step 1: create an app and define the service endpoint
        app = FastAPI(title='Custom Gateway')

        @app.get(path='/generate')
        def generate():
            max_length = 100
            num_return_sequences = 1
            input_text = "Translate the following English text to French: 'Hey, how are you?'"

            input_ids = self.tokenizer.encode(input_text, return_tensors="pt")

            # Move input to the first GPU
            input_ids = input_ids.to("cuda:0")

            # Generate output with the model
            outputs = self.model.generate(
                input_ids,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
            )

            # Move generated output to the CPU
            outputs = outputs.cpu()

            # Decode the outputs to obtain text
            generated_text = [
                self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs
            ]

            return {'message': generated_text}

        # step 5: implement health-check
        @app.get(path='/')
        def health_check():
            return {}

        class UviServer(Server):
            """The uvicorn server."""

            async def setup(self, sockets=None):
                """
                Setup uvicorn server.

                :param sockets: sockets of server.
                """
                config = self.config
                if not config.loaded:
                    config.load()
                self.lifespan = config.lifespan_class(config)
                await self.startup(sockets=sockets)
                if self.should_exit:
                    return

            async def serve(self, **kwargs):
                """
                Start the server.

                :param kwargs: keyword arguments
                """
                await self.main_loop()

        # step 6: bind the gateway server to the right port and host
        print(self.port, self.host)
        self.server = UviServer(Config(app, host=self.host, port=self.port))
        await self.server.setup()

    async def shutdown(self):
        """
        Free resources allocated when setting up HTTP server
        """
        self.server.should_exit = True
        await self.server.shutdown()

    async def run_server(self):
        """Run HTTP server forever"""
        await self.server.serve()


flow = Flow().config_gateway(
    uses=MyGateway, port=12348, protocol='http', timeout_ready=-1, uses_with={
        'model_name': "google/flan-t5-xl"
    }
).add(
    uses=Encoder, uses_with={
        'model_name': "google/flan-t5-xl",
        'device_map': {
            0: list(range(0, 8)),
            1: list(range(8, 16)),
            2: list(range(16, 24)),
        }}, port=12346
).add(
    uses=Decoder, uses_with={
        'model_name': "google/flan-t5-xl",
        'device_map': {
            0: list(range(0, 8)),
            1: list(range(8, 16)),
            2: list(range(16, 24)),
        }}, port=12347
)
with flow as f:
    f.block()
