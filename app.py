from flask import (
    Flask,
    render_template
)

import os
app = Flask(__name__)
app.config["DEBUG"] = True

@app.route('/', methods=['GET'])
def home():
    return render_template('form.html')

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))