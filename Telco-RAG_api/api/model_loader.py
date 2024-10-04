from transformers import AutoModelForCausalLM, AutoTokenizer

class ModelLoader:
    _model = None
    _tokenizer = None

    @staticmethod
    def load_model(model_name):
        if ModelLoader._model is None or ModelLoader._tokenizer is None:
            ModelLoader._tokenizer = AutoTokenizer.from_pretrained(model_name)
            ModelLoader._model = AutoModelForCausalLM.from_pretrained(model_name)
        return ModelLoader._model, ModelLoader._tokenizer
