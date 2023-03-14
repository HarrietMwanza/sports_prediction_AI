from flask import (
    Flask,
    render_template,
    request
)

from sports_prediction import predict_rating

import os
app = Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('form.html')


@app.route('/ratings', methods=['POST'])
def ratings():
    # get values from submitted form: previous_rank_points, ranking_change

    previous_rank_points = request.form['previous_rank_points']
    ranking_change = request.form['ranking_change']

    print("previous_rank_points:", previous_rank_points)
    print("ranking_change:", ranking_change)

    return predict_rating(previous_rank_points, ranking_change)


if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
