import dash
from dash import html, dcc, State
from dash.dependencies import Input, Output
import torch
import json

from model import LSTMLanguageModel



# Load vocabulary


with open("vocab.json", "r", encoding="utf-8") as f:
    stoi = json.load(f)

itos = {idx: token for token, idx in stoi.items()}


class SimpleVocab:
    def __init__(self, stoi, itos):
        self.stoi = stoi
        self.itos = itos

    def __len__(self):
        return len(self.stoi)

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi.get("<unk>"))

    def lookup_token(self, index):
        return self.itos.get(index, "<unk>")


vocab = SimpleVocab(stoi, itos)


# Load model


device = torch.device("cpu")

VOCAB_SIZE = len(vocab)
EMBEDDING_DIM = 1024   # MUST match training
HIDDEN_DIM = 1024      # MUST match training
NUM_LAYERS = 2
DROPOUT = 0.65

model = LSTMLanguageModel(
    VOCAB_SIZE,
    EMBEDDING_DIM,
    HIDDEN_DIM,
    NUM_LAYERS,
    DROPOUT
).to(device)

model.load_state_dict(
    torch.load("best-val-lstm_lm.pt", map_location=device)
)
model.eval()



# Sampling function 

def sample_next_word(probs, temperature=0.8):
    probs = torch.log(probs + 1e-9) / temperature
    probs = torch.softmax(probs, dim=0)
    return torch.multinomial(probs, 1).item()



# Text generation

def generate_continuation(prompt, max_len=30, temperature=0.8):
    model.eval()

    words = prompt.lower().split()

    indices = [
        vocab[w] if w in vocab.stoi else vocab["<unk>"]
        for w in words
    ]

    input_tensor = torch.LongTensor(indices).unsqueeze(1).to(device)
    hidden = model.init_hidden(1)

    generated_words = []

    with torch.no_grad():
        for _ in range(max_len):
            output, hidden = model(input_tensor, hidden)

            probs = torch.softmax(output[-1], dim=0)
            next_idx = sample_next_word(probs, temperature)

            next_word = vocab.lookup_token(next_idx)

            if next_word == "<eos>":
                break

            generated_words.append(next_word)

            input_tensor = torch.LongTensor([[next_idx]]).to(device)

    return " ".join(generated_words)



# Dash App

app = dash.Dash(__name__)

app.layout = html.Div(
    style={
        "width": "60%",
        "margin": "auto",
        "fontFamily": "Arial"
    },
    children=[
        html.H1("Harry Potter LSTM Language Model"),

        dcc.Textarea(
            id="input-text",
            placeholder="Type a prompt (e.g., 'harry potter is')",
            style={"width": "100%", "height": "100px"}
        ),

        html.Br(),

        html.Button("Generate Continuation", id="generate-btn"),

        html.Br(), html.Br(),

        html.Div(
            id="output-text",
            style={
                "whiteSpace": "pre-wrap",
                "border": "1px solid #ccc",
                "padding": "10px",
                "minHeight": "80px"
            }
        )
    ]
)


# Callback

@app.callback(
    Output("output-text", "children"),
    Input("generate-btn", "n_clicks"),
    State("input-text", "value")
)
def generate_callback(n_clicks, prompt):
    if not n_clicks or not prompt:
        return ""

    continuation = generate_continuation(
        prompt,
        max_len=30,
        temperature=0.8
    )

    return continuation


# Run

if __name__ == "__main__":
    app.run(debug=True)
