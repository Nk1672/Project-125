from flask import Flask,jsonify,request
from classifier import get_pred

app = Flask(__name__)

@app.route('/predict-digit',methods=['POST'])
def predict_data():
    img=request.files.get('digit')
    prediction=get_pred(img)
    return jsonify({
        'predictions':prediction
        },200)
    
if (__name__=='__main__'):
    app.run(debug=True)
