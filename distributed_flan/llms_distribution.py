import io

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
import random
import numpy as np
from transformers.models.t5.modeling_t5 import T5Stack
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from jina import Flow, Client
from fastapi import FastAPI
from uvicorn import Server, Config
from jina import DocumentArray, Gateway
from executor.encoder import EncoderExecutor, InputSchema, OutputSchema
from executor.decoder import DecoderExecutor


def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


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
            inputs=DocumentArray(
                [
                    InputSchema(
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
                    )
                ]
            ),
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
        return BaseModelOutputWithPastAndCrossAttentions(**outputs)


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
            'cuda:0'
        )
        encoder_client = Client(port=12346)
        decoder_client = Client(port=12347)
        self.model.encoder = T5StackWrapper(
            config, encoder_client, embed_tokens=embedding_layer
        )

        self.model.decoder = T5StackWrapper(
            config, decoder_client, embed_tokens=embedding_layer
        )
        self.model.lm_head = self.model.lm_head.to('cuda:0')
        self.model.parallel = True

    async def setup_server(self):
        # step 1: create an app and define the service endpoint
        app = FastAPI(title='Custom Gateway')

        from pydantic import BaseModel

        class Input(BaseModel):
            input_text: str

        @app.get(path='/generate')
        def generate(prompt: Input):
            max_length = 100
            num_return_sequences = 1
            input_text = prompt.input_text

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
                self.tokenizer.decode(output, skip_special_tokens=True)
                for output in outputs
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


flow = (
    Flow()
    .config_gateway(
        uses=MyGateway,
        port=12348,
        protocol='http',
        timeout_ready=-1,
        uses_with={'model_name': "google/flan-t5-xl"},
    )
    .add(
        uses=EncoderExecutor,
        uses_with={
            'model_name': "google/flan-t5-xl",
            'device_map': {
                0: list(range(0, 8)),
                1: list(range(8, 16)),
                2: list(range(16, 24)),
            },
        },
        port=12346,
    )
    .add(
        uses=DecoderExecutor,
        uses_with={
            'model_name': "google/flan-t5-xl",
            'device_map': {
                0: list(range(0, 8)),
                1: list(range(8, 16)),
                2: list(range(16, 24)),
            },
        },
        port=12347,
    )
)
with flow as f:
    f.block()


import requests

# Set up the payload with the prompt as a query parameter
payload = {
    'input_text': "Translate the following English text to French: 'Hey, how are you?'"
}

# Make the request
response = requests.get('http://localhost://12348/generate', params=payload)
