// frontend/script.js

const form = document.getElementById("predictForm");
const resultBox = document.getElementById("userResult");

form.addEventListener("submit", async (e) => {
    e.preventDefault();

    // =========================
    // Build request payload
    // =========================
    const formData = new FormData(form);
    const payload = {};

    for (const [key, value] of formData.entries()) {
        const v = value.trim();
        payload[key] = v === "" ? 0 : parseFloat(v);
    }

    // Loading state
    resultBox.innerHTML = `
        <div class="empty-result">
            <div class="result-icon">...</div>
            <h2>Analyzing Student</h2>
            <p>Please wait while the model generates a prediction.</p>
        </div>
    `;

    try {
        // =========================
        // Send request to FastAPI
        // =========================
        const res = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        });

        if (!res.ok) {
            resultBox.innerHTML = `
                <div class="empty-result">
                    <div class="result-icon">!</div>
                    <h2>Prediction Failed</h2>
                    <p>
                        Server returned ${res.status} ${res.statusText}.
                    </p>
                </div>
            `;

            return;
        }

        // =========================
        // Read prediction
        // =========================
        const data = await res.json();

        const probability = data.risk_probability;
        const probPercent = (probability * 100).toFixed(2);

        const isAtRisk = data.prediction === 1;

        const predictionText = isAtRisk
            ? "At Risk"
            : "Not At Risk";

        const resultClass = isAtRisk
            ? "at-risk"
            : "not-at-risk";

        // =========================
        // Display result
        // =========================
        resultBox.innerHTML = `
            <div class="prediction-result ${resultClass}">

                <div class="prediction-label">
                    ${predictionText}
                </div>

                <h2>Student Risk Assessment</h2>

                <p>Estimated probability of being at risk</p>

                <div class="probability-value">
                    ${probPercent}%
                </div>

                <div class="progress-container">
                    <div
                        class="progress-bar"
                        style="width: ${probPercent}%">
                    </div>
                </div>

                <p>
                    Decision threshold: 50%
                </p>

            </div>
        `;

    } catch (err) {
        console.error(err);

        resultBox.innerHTML = `
            <div class="empty-result">
                <div class="result-icon">!</div>
                <h2>Connection Error</h2>
                <p>
                    Unable to connect to the prediction API.
                    Please make sure the backend is running.
                </p>
            </div>
        `;
    }
});