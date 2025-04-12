from flask import Flask, render_template, request
from pipeline.prediction_pipeline import hybrid_recommendation
from utils.helpers import get_random_anime

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = None
    if request.method == "POST":
        try:
            user_id = int(request.form["userID"])
            recommendations_df = hybrid_recommendation(user_id)

            # if no valid recommendations returned
            if recommendations_df is None or recommendations_df.empty:
                raise ValueError("No recommendations for this user.")

            # Get anime names only
            recommendations = recommendations_df["anime_name"].tolist()

        except Exception as e:
            print("Error occurred:", e)
            # fallback to 10 random anime titles
            recommendations = get_random_anime(10)

    return render_template("index.html", recommendations=recommendations)

if __name__=="__main__":
    app.run(debug=True,host='0.0.0.0',port=5000)