const inferBtn = document.getElementById("inferBtn");
const metricsBtn = document.getElementById("metricsBtn");
const hideMetricsBtn = document.getElementById("hideMetricsBtn");

const selectedFile = document.getElementById("selectedFile");
const fileInput = document.getElementById("file");
const modelSelect = document.getElementById("modelSelect");

const loading = document.getElementById("loading");

const metricsContainer = document.getElementById("metricsContainer");
const metricsGrid = document.getElementById("metricsGrid");

const overlayImg = document.getElementById("overlayImg");
const confImg = document.getElementById("confImg");
const entropyImg = document.getElementById("entropyImg");

const downloadOverlayBtn = document.getElementById("downloadOverlayBtn");
const downloadConfBtn = document.getElementById("downloadConfBtn");
const downloadEntropyBtn = document.getElementById("downloadEntropyBtn");

fileInput.addEventListener("change", () => {
    selectedFile.textContent =
        fileInput.files.length ? fileInput.files[0].name : "No image selected";
});

inferBtn.addEventListener("click", async () => {
    const file = fileInput.files[0];
    if (!file) {
        alert("Select an image.");
        return;
    }

    loading.style.display = "block";
    const formData = new FormData();
    formData.append("file", file);
    formData.append("model_name", modelSelect.value);
    try {
        const response = await fetch("/predict", {
            method: "POST",
            
            body: formData
        });
        const data = await response.json();
        if (data.error) {
            throw new Error(data.error);
        }

        overlayImg.src = `data:image/png;base64,${data.overlay}`;
        confImg.src = `data:image/png;base64,${data.confidence}`;
        entropyImg.src = `data:image/png;base64,${data.entropy}`;
        downloadOverlayBtn.style.display = "inline-block";
        downloadConfBtn.style.display = "inline-block";
        downloadEntropyBtn.style.display = "inline-block";
    } catch (err) {
        alert(err.message);
    } finally {
        loading.style.display = "none";
    }
});

metricsBtn.addEventListener("click", async () => {
    try {
        const response = await fetch("/metrics");
        const data = await response.json();
        renderMetrics(data);

        metricsContainer.style.display = "block";
        metricsBtn.style.display = "none";
        hideMetricsBtn.style.display = "inline-block";
    } catch (err) {
        alert("Metrics error");
    }
});

hideMetricsBtn.addEventListener("click", () => {
    metricsContainer.style.display = "none";
    hideMetricsBtn.style.display = "none";
    metricsBtn.style.display = "inline-block";
});

[   [downloadOverlayBtn, overlayImg, "overlay.png"],
    [downloadConfBtn, confImg, "confidence.png"],
    [downloadEntropyBtn, entropyImg, "entropy.png"]
].forEach(([button, image, filename]) => {
    button.addEventListener("click", () => downloadImage(image, filename));
});


function renderMetrics(data) {
    metricsGrid.innerHTML = "";
    for (const key in data) {
        const card = document.createElement("div");
        card.className = "metric-card";
        const value = Array.isArray(data[key]) ? data[key].join(" x ") : data[key];
        card.innerHTML = `
            <span class="metric-name">
                ${key}
            </span>
            <span class="metric-value">
                ${value}
            </span>
        `;
        metricsGrid.appendChild(card);
    }
}

function downloadImage(imgElement, filename) {
    const a = document.createElement("a");
    a.href = imgElement.src;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}
