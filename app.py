import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load the dataset
sonar_data = pd.read_csv(r"C:\Users\dell\OneDrive\Desktop\Rock&mine\copy.csv", header=None)

# Prepare the data and model
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60].map({'R': 0, 'M': 1})

# Train the best model (Random Forest)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_scaled, Y)

class RequestHandler(BaseHTTPRequestHandler):
    def _send_response(self, status_code, response_body):
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response_body).encode('utf-8'))

    def do_POST(self):
        if self.path == '/predict':
            # Parse the incoming data
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data)
                features = list(map(float, data['features']))

                if len(features) != 60:
                    self._send_response(400, {"error": "Please provide exactly 60 features."})
                    return

                # Preprocess the input features and make a prediction
                input_data = np.array(features).reshape(1, -1)
                input_data_scaled = scaler.transform(input_data)
                prediction = model.predict(input_data_scaled)
                prediction_label = 'Rock' if prediction[0] == 0 else 'Mine'

                # Generate the confusion matrix
                Y_pred = model.predict(X_scaled)
                conf_matrix = confusion_matrix(Y, Y_pred)
                cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["Rock", "Mine"])

                # Save the confusion matrix as an image
                cm_img_path = 'confusion_matrix.png'
                cm_display.plot(cmap='Blues')
                plt.savefig(cm_img_path)
                plt.close()

                # Respond with the prediction and image path
                self._send_response(200, {
                    'prediction': prediction_label,
                    'confusion_matrix_img': cm_img_path
                })

            except Exception as e:
                self._send_response(500, {"error": str(e)})

# Run the HTTP server
def run(server_class=HTTPServer, handler_class=RequestHandler, port=8000):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f"Server running on port {port}...")
    httpd.serve_forever()

if __name__ == '__main__':
    run()