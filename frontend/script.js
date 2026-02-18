async function predict() {

    const input = document.getElementById("features").value;

    const features = input.split(",").map(Number);

    if (features.length !== 30) {
        alert("You must enter exactly 30 features");
        return;
    }

    const response = await fetch("https://phishing-detector-r5ff.onrender.com/predict", {

        method: "POST",

        headers: {
            "Content-Type": "application/json"
        },

        body: JSON.stringify({
            features: features
        })

    });

    const result = await response.json();

    document.getElementById("result").innerHTML =
        `
        <h3>Prediction: ${result.prediction}</h3>
        <h3>Probability: ${result.probability.toFixed(6)}</h3>
        `;
}
