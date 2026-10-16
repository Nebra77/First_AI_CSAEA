import torch
from flask import Flask, jsonify, send_file
 
from main import GPTLanguageModel, decode, device
 
app = Flask(__name__)
 
model = GPTLanguageModel().to(device)
model.load_state_dict(torch.load('model.pt', map_location=device))
model.eval()
 
@app.route('/')
def home():
    return send_file('index.html')
 
@app.route('/generate')
def generate():
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    with torch.no_grad():
        tokens = model.generate(context, max_new_tokens=500)[0].tolist()
    return jsonify(text=decode(tokens))
 

if __name__ == '__main__':
    app.run(debug=False)
 