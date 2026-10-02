// frontend/script.js
const form = document.getElementById("predictForm");
const resultBox = document.getElementById("userResult");

form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const formData = new FormData(form);
    const payload = {};

    for (const [key, value] of formData.entries()) {
        const v = value.trim();
        payload[key] = v === "" ? 0 : parseFloat(v);
    }

    resultBox.textContent = "Sending request...";

    try {
        const res = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(payload)
        });

        if (!res.ok) {
            resultBox.textContent = "Error: " + res.status + " " + res.statusText;
            return;
        }

        const data = await res.json();

        const probPercent = (data.risk_probability * 100).toFixed(2);

        const predictionText = data.prediction === 1 ? "At Risk" : "Not At Risk";

        resultBox.innerHTML =
            `<b>Prediction: ${predictionText}</b><br>` +
            `At-Risk Probability: ${probPercent}%`;
    } catch (err) {
        console.error(err);
        resultBox.textContent = "Request failed. Please check backend.";
    }
});
