// Function to handle the input prediction and display the result
function predict() {
    var inputData = document.getElementById('inputData').value;
    
    // Basic validation for input data (it should be 60 comma-separated values)
    var features = inputData.split(',');
    if (features.length !== 60) {
        alert("Error: Please enter exactly 60 features.");
        return;
    }

    // Send the data to the server (Flask) via AJAX (using Fetch API)
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ features: features })
    })
    .then(response => response.json())
    .then(data => {
        // Display the result of the prediction
        document.getElementById('result').innerText = data.prediction;

        // Display Confusion Matrix image (if available)
        if (data.confusion_matrix_img) {
            var img = document.getElementById('confusionMatrixImg');
            img.src = data.confusion_matrix_img;
            document.getElementById('confusionMatrixContainer').style.display = 'block';
        }
    })
    .catch(error => {
        alert('Error: ' + error);
    });
}
