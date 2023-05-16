from transformers import AutoTokenizer, AutoModel
import warnings, logging
warnings.simplefilter('ignore')
logging.disable(logging.WARNING)

# Load model from HuggingFace Hub
model_name = 'all-mpnet-base-v2'
tokenizer = AutoTokenizer.from_pretrained(f'sentence-transformers/{model_name}')
model = AutoModel.from_pretrained(f'sentence-transformers/{model_name}')
save_path = f'./models/{model_name}'

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)


