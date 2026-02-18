async function predictURL() {

    const url = document.getElementById("url").value;

    if (!url) {
        alert("Please enter a URL");
        return;
    }

    const response = await fetch("https://phishing-detector-r5ff.onrender.com/predict-url", {

        method: "POST",

        headers: {
            "Content-Type": "application/json"
        },

        body: JSON.stringify({
            url: url
        })

    });

    const result = await response.json();

    document.getElementById("result").innerHTML =
        `
        <h3>Prediction: ${result.prediction}</h3>
        <h3>Probability: ${result.probability.toFixed(4)}</h3>
        `;
}
