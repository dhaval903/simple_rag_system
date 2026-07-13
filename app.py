from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify

from rag import ask_question

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/chat',methods = ["POST"])
def chat():
    data = request.json
    
    question = data["message"]
    
    result = ask_question(question)
    
    return jsonify(result)

if __name__ = "__main__":
    app.run(debug=True)