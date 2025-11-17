from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("ac_power_model.pkl")

@app.route("/")
def home():
    return render_template("index.html",
                           irradiation="",
                           ambient_temp="",
                           module_temp="",
                           dc_power="",
                           hour="",
                           day="",
                           month="",
                           prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        irradiation = request.form.get("irradiation", "")
        ambient_temp = request.form.get("ambient_temp", "")
        module_temp = request.form.get("module_temp", "")
        dc_power = request.form.get("dc_power", "")
        hour = request.form.get("hour", "")
        day = request.form.get("day", "")
        month = request.form.get("month", "")

        features = np.array([[float(irradiation),
                              float(ambient_temp),
                              float(module_temp),
                              float(dc_power),
                              int(hour),
                              int(day),
                              int(month)]])

        prediction = model.predict(features)[0]

        return render_template("index.html",
                               irradiation=irradiation,
                               ambient_temp=ambient_temp,
                               module_temp=module_temp,
                               dc_power=dc_power,
                               hour=hour,
                               day=day,
                               month=month,
                               prediction=f"{prediction:.2f}")

    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    app.run(debug=True)
