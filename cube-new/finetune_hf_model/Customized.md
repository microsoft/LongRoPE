## Customized function

Register a customized function means take the function as a leaf node in the model graph,
this means the function will not be split into fine-grained ops,
thus ensuring the full functionality of the customized function.

About the detail of `cube.graph.parser.register`, please view <>.

## Customized model

This example has provided support for the models that can be loaded by:

* transformers.AutoModel.from_pretrained
* transformers.AutoModelForCausalLM.from_pretrained
* transformers.AutoModelForSeq2SeqLM.from_pretrained
* transformers.AutoModelForSequenceClassification.from_pretrained
* transformers.AutoModelForTokenClassification.from_pretrained
* transformers.AutoModelForQuestionAnswering.from_pretrained

For a specific customized model, you can register it as a fairseq model to support it.
