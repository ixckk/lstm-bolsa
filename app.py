from flask import Flask, jsonify, render_template, request

from model import Config, run

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/predict")
def predict():
    body = request.get_json(force=True) or {}
    try:
        cfg = Config(
            ticker=str(body.get("ticker", "AAPL")).strip().upper()[:15],
            period=body.get("period", "5y") if body.get("period") in {"2y", "5y", "10y", "max"} else "5y",
            window=max(5, min(int(body.get("window", 30)), 120)),
            epochs=max(1, min(int(body.get("epochs", 40)), 200)),
            hidden=max(8, min(int(body.get("hidden", 64)), 256)),
            horizon=max(1, min(int(body.get("horizon", 10)), 60)),
        )
        return jsonify(run(cfg))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"Error interno: {e}"}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)
