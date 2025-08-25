import torch
from torch.nn import Embedding, LSTM, Linear, Dropout, LayerNorm, GELU, LogSoftmax
from transformers import AutoTokenizer
import time
import os


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

        self.labelMap = {0: 'Happiness',
                         1: 'Neutral',
                         2: 'Sadness',
                         3: 'Surprise',
                         4: 'Love',
                         5: 'Fear',
                         6: 'Confusion',
                         7: 'Disgust',
                         8: 'Desire',
                         9: 'Shame',
                         10: 'Sarcasm',
                         11: 'Anger',
                         12: 'Guilt'}

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

    def getMap(self):
        return self.labelMap


def inference(model, text, tokenizer):
    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt"
    )

    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']

    output = model(input_ids=input_ids, attention_mask=attention_mask)

    return output.argmax(dim=1).item()



try:
    # Load the tokenizer
    tokenizer_path = "../../model/tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
except Exception as e:
    print(f"Error loading tokenizer: {e}")


#Create instance of model
txtEmotionModel = EmotionBiLSTM(28996, 275, 64, 13)

# Load the weights

try:

    txtEmotionModel.load_state_dict(torch.load("../../model/emotion_model.pt"))
    txtEmotionModel.eval()  # Set to evaluation mode

except Exception as e:
    print(f"Error loading model weights: {e}")







