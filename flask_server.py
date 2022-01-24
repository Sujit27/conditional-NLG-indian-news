import os
from flask import Flask, json, request
from flask_cors import CORS, cross_origin
from generate import get_tokenier,generate_text,get_model,DEVICE,SPECIAL_TOKENS,MAXLEN,MODEL

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

tokenizer = get_tokenier(special_tokens=SPECIAL_TOKENS)
model = get_model(tokenizer, special_tokens=SPECIAL_TOKENS,load_model_path=os.path.join(MODEL,'pytorch_model.bin'))


@app.route('/generate_text', methods=['POST'])
@cross_origin()
def gen_text():
    body = request.get_json()
    input_headline = str(body["headline"])
    input_keystring = str(body["keystring"])
    try:
        input_keywords = body["keywords"]
        print(input_keywords)
    except:
        input_keywords = str(body["keystring"]).split(",")
    
    output = generate_text(input_headline,input_keywords,model,tokenizer)
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    strip_len = len(input_headline) + len(','.join(input_keywords))  
    generated_text =  text[strip_len:]


    if(generated_text):
        return json.dumps({'generated_text': str(generated_text)})
    else:
        return json.dumps({'generate_text': 'false'})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=True)