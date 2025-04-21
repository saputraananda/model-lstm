import logging
import time
from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
import joblib
from keras.models import load_model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

app = Flask(__name__)

# Load model dan scaler
try:
    logging.info("Loading model and scaler...")
    model = load_model('model_penjualan.h5', compile=False)
    scaler = joblib.load('scaler_penjualan.pkl')
    logging.info("Model and scaler loaded successfully.")
except Exception as e:
    logging.error("Failed to load model or scaler: %s", str(e))
    raise e

@app.route('/')
def home():
    return jsonify({"status": "API Flask untuk Prediksi Penjualan aktif"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    logging.info("🚀 Memulai proses prediksi...")

    try:
        data = request.get_json()
        last_sequence = np.array(data['last_sequence'])
        last_date = pd.to_datetime(data.get('last_date'))

        if len(last_sequence.shape) != 2 or last_sequence.shape[1] != 4:
            return jsonify({'error': 'Format last_sequence tidak sesuai'}), 400

        logging.info("✅ Last sequence shape: %s", last_sequence.shape)
        logging.info("🗓️  Last date: %s", last_date)

        n_past = last_sequence.shape[0]
        features = last_sequence.shape[1]
        current_seq = last_sequence.reshape(1, n_past, features)

        future_predictions = []
        prediction_dates = []

        for i in range(150):  # prediksi 30 hari ke depan
            next_pred = model.predict(current_seq, verbose=0)[0][0]
            future_predictions.append(next_pred)

            next_date = last_date + pd.Timedelta(days=i + 1)
            prediction_dates.append(next_date.strftime('%Y-%m-%d'))

            day, month = next_date.day, next_date.month
            is_weekend = int(next_date.weekday() >= 5)
            next_features = np.array([[next_pred, day, month, is_weekend]])

            current_seq = np.append(
                current_seq[:, 1:, :],
                next_features.reshape(1, 1, features),
                axis=1
            )

        # Inverse scaling
        result = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten().tolist()

        # Bagi hasil
        forecast_5hari = result[:5]
        forecast_5minggu = [result[i] for i in [6, 13, 20, 27, 29]]  # prediksi minggu ke-1 sampai 5
        forecast_5bulan = result  # semua 30 hari

        sim_mape = np.mean(np.abs((np.array(result[1:]) - np.array(result[:-1])) / np.array(result[:-1]))) * 100
        sim_mape = round(sim_mape, 2)

        logging.info("✅ Prediksi selesai. Simulasi MAPE: %.2f%%", sim_mape)
        logging.info("🕒 Durasi proses prediksi: %.2fs", time.time() - start_time)

        return jsonify({
            'prediksi': result,  # ✅ Tambahkan baris ini untuk Laravel
            'tanggal': prediction_dates,
            'mape_simulasi (%)': sim_mape
        })


    except Exception as e:
        logging.exception("Gagal memproses prediksi")
        return jsonify({'error': str(e)}), 500

@app.route('/evaluate', methods=['POST'])
def evaluate():
    start_time = time.time()
    logging.info("Mulai proses evaluasi...")

    try:
        data = request.get_json()
        hist_seq = np.array(data['historical_sequence'])
        actual = np.array(data['actual_values'])

        logging.info("Jumlah data historis: %d", len(hist_seq))
        logging.info("Jumlah data aktual: %d", len(actual))

        n_past = hist_seq.shape[0]
        features = hist_seq.shape[1]
        current_seq = hist_seq.reshape(1, n_past, features)

        future_predictions = []
        last_date = pd.Timestamp.today()

        for i in range(len(actual)):
            next_pred = model.predict(current_seq, verbose=0)[0][0]
            future_predictions.append(next_pred)

            next_date = last_date + pd.Timedelta(days=i+1)
            day, month = next_date.day, next_date.month
            is_weekend = int(next_date.weekday() >= 5)
            next_features = np.array([[next_pred, day, month, is_weekend]])

            current_seq = np.append(current_seq[:, 1:, :], next_features.reshape(1, 1, features), axis=1)

        pred_unscaled = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()
        mape = np.mean(np.abs((actual - pred_unscaled) / actual)) * 100
        mape = round(mape, 2)

        logging.info("Evaluasi selesai. MAPE: %.2f%%", mape)
        logging.info("Durasi proses evaluasi: %.2fs", time.time() - start_time)

        return jsonify({
            'actual': actual.tolist(),
            'prediksi': pred_unscaled.tolist(),
            'mape': mape
        })

    except Exception as e:
        logging.exception("Gagal memproses evaluasi")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logging.info("Menjalankan API Flask di http://0.0.0.0:5000 ...")
    app.run(debug=True, host='0.0.0.0', port=5000)
