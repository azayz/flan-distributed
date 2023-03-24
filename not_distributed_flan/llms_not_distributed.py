from docarray import DocumentArray
from jina import Flow
from executor import FlanExecutor, InputSchema, OutputSchema


f = Flow().add(
    uses=FlanExecutor,
    uses_with={
        'model_name': "google/flan-t5-xl",
    },
)

with f:
    docs = f.post(
        on='/',
        inputs=DocumentArray(
            [
                InputSchema(
                    text="Translate the following English text to French: 'Hey, how are you?'"
                )
            ]
        ),
        return_type=DocumentArray[OutputSchema],
    )
    outputs = dict(docs[0])
    print(outputs)
    docs.summary()
