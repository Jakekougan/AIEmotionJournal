import torch
from torch.nn import Embedding, LSTM, Linear, Dropout, LayerNorm, GELU, LogSoftmax
from transformers import AutoTokenizer
import time


#custom model
class EmotionBiLSTM(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_labels):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
        self.lstm = torch.nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True, num_layers=2)
        self.dropout = torch.nn.Dropout(0.5)
        self.norm = torch.nn.LayerNorm(hidden_dim * 2)
        self.fc1 = torch.nn.Linear(hidden_dim * 2, 64)
        self.fc2 = torch.nn.Linear(64, num_labels)

        self.act = torch.nn.GELU()



    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)

        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).cpu()
            x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        _, (hn, _) = self.lstm(x)

        # Concatenate final hidden states from both directions
        x = torch.cat((hn[-2], hn[-1]), dim=1)

        x = self.norm(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = LogSoftmax(dim=1)(x)


        return x



def inference(model, text, tokenizer):
    encoded = tokenizer(
    text.tolist(),
    padding="max_length",
    truncation=True,
    max_length=64,
    return_tensors="pt"
    )

    input_ids = encoded['input_ids'].tolist()
    attention_mask = encoded['attention_mask'].tolist()

    output = model(input_ids=input_ids, attention_mask=attention_mask)


    return labelMap.get(output.argmax(dim=1).item(), "Unknown Emotion")

labelMap = {0: 'happiness',
            1: 'neutral',
            2: 'sadness',
            3: 'surprise',
            4: 'love',
            5: 'fear',
            6: 'confusion',
            7: 'disgust',
            8: 'desire',
            9: 'shame',
            10: 'sarcasm',
            11: 'anger',
            12: 'guilt'}



print("Downloading model...")

try:
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
except Exception as e:
    print(f"Error loading tokenizer: {e}")





#Create instance of model
model = EmotionBiLSTM(28732, 250, 64, 13)

# Load the weights

try:

    model.load_state_dict(torch.load("model\\emotion_model.pt"))
    model.eval()  # Set to evaluation mode

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    print("Model Download Successful!")

except Exception as e:
    print(f"Error loading model weights: {e}")









