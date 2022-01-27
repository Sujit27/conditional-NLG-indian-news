import os
from flask import Flask, json, request
from flask_cors import CORS, cross_origin
from generate import get_tokenier,generate_text,get_model,DEVICE,SPECIAL_TOKENS,MAXLEN,MODEL

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

print("Loading models...")
tokenizer = get_tokenier(special_tokens=SPECIAL_TOKENS)
model = get_model(tokenizer, special_tokens=SPECIAL_TOKENS,load_model_path=os.path.join(MODEL,'pytorch_model.bin'))
print("Models loaded")


@app.route('/generate_text', methods=['POST'])
@cross_origin()
def gen_text():
    body = request.get_json()
    input_headline = body["headline"]
    input_keywords = body["keywords"]
    if isinstance(input_keywords, str): 
        input_keywords = input_keywords.split(",")
    
    print("Generating text...")
    output = generate_text(input_headline,input_keywords,model,tokenizer)
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    strip_len = len(input_headline) + len(','.join(input_keywords))  
    generated_text =  text[strip_len:]


    if(generated_text):
        return json.dumps({'generated_text': str(generated_text)})
    else:
        return json.dumps({'generate_text': 'false'})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=True,use_reloader=False)