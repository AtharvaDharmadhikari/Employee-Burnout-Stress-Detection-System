/* Webcam emotion detection for Flask check-in page */

let stream = null;
let detectedMood = null;
let detectedStress = null;

async function startCamera(videoId) {
    const video = document.getElementById(videoId);
    if (!video) return;
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        video.srcObject = stream;
    } catch (err) {
        showCameraError("Camera access denied. Please allow camera permission or use manual mode.");
    }
}

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(t => t.stop());
        stream = null;
    }
}

function captureAndDetect(videoId, canvasId, action) {
    const video  = document.getElementById(videoId);
    const canvas = document.getElementById(canvasId);
    if (!video || !canvas || !stream) {
        showCameraError("Camera not ready. Please wait a moment.");
        return;
    }

    canvas.width  = video.videoWidth  || 640;
    canvas.height = video.videoHeight || 480;
    canvas.getContext("2d").drawImage(video, 0, 0);
    const imageData = canvas.toDataURL("image/jpeg", 0.85);

    const btn = document.getElementById("captureBtn" + (action === "checkin" ? "In" : "Out"));
    if (btn) { btn.disabled = true; btn.textContent = "Detecting…"; }

    showDetecting();

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 60000); // 60s timeout

    fetch("/api/detect-emotion", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: imageData }),
        signal: controller.signal,
    })
    .then(r => { clearTimeout(timeout); return r.json(); })
    .then(data => {
        if (btn) { btn.disabled = false; btn.innerHTML = '<i class="bi bi-camera"></i> Capture Again'; }
        if (data.success) {
            detectedMood   = data.mood;
            detectedStress = data.stress_level;
            showResult(data, action);
        } else {
            showCameraError("Detection failed: " + (data.error || "Unknown error"));
        }
    })
    .catch(err => {
        clearTimeout(timeout);
        if (btn) { btn.disabled = false; btn.textContent = "Capture"; }
        if (err.name === "AbortError") {
            showCameraError("Detection timed out (60s). The server may still be loading AI models — try again in a few seconds.");
        } else {
            showCameraError("Network error: " + err.message + ". Check that Flask server is running on port 5000.");
        }
    });
}

function showDetecting() {
    const el = document.getElementById("detectionResult");
    if (el) {
        el.innerHTML = `
            <div class="text-center py-3">
                <div class="spinner-border text-primary" role="status"></div>
                <p class="mt-2 text-muted">Analysing your expression…</p>
            </div>`;
        el.style.display = "block";
    }
}

const MOOD_EMOJI = {
    Happy: "😊", Calm: "😌", Neutral: "😐", Tired: "😴",
    Sad: "😢", Stressed: "😰", Angry: "😠",
    Surprised: "😲", Fear: "😨", Disgust: "🤢",
    Happy_default: "😊"
};

function showResult(data, action) {
    const el = document.getElementById("detectionResult");
    if (!el) return;
    const emoji = MOOD_EMOJI[data.mood] || "😐";
    const label = action === "checkin" ? "Confirm Check-In" : "Confirm Check-Out";
    const stressColor = data.stress_level >= 7 ? "#ef4444" : data.stress_level >= 5 ? "#f59e0b" : "#10b981";

    el.innerHTML = `
        <div class="mood-result-card">
            <div class="mood-result-emoji">${emoji}</div>
            <div class="mood-result-label">${data.mood}</div>
            <div class="mood-result-stress">
                Stress Level: <strong>${data.stress_level}/10</strong>
                <span style="color:${stressColor}; margin-left:6px">●</span>
            </div>
            ${data.confidence ? `<div style="font-size:0.76rem;opacity:0.7;margin-top:4px">Confidence: ${data.confidence}%</div>` : ''}
        </div>
        <button onclick="confirmSave('${action}')" class="btn-primary-custom w-100">
            <i class="bi bi-check-circle"></i> ${label}
        </button>`;
    el.style.display = "block";
}

function showCameraError(msg) {
    const el = document.getElementById("detectionResult");
    if (el) {
        el.innerHTML = `<div class="alert alert-warning">${msg}</div>`;
        el.style.display = "block";
    }
}

function confirmSave(action) {
    if (!detectedMood) return;
    const btn = event.target;
    btn.disabled = true;
    btn.textContent = "Saving…";

    fetch("/api/save-emotion", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            mood: detectedMood,
            stress_level: detectedStress,
            source: "webcam",
            action: action
        })
    })
    .then(r => r.json())
    .then(data => {
        if (data.success) {
            stopCamera();
            const el = document.getElementById("detectionResult");
            let extra = "";
            if (data.recommendation) {
                extra = `<p class="mt-2 mb-0" style="font-size:0.85rem">
                    <strong>Suggested task:</strong> ${data.recommendation} — ${data.description || ""}
                </p>`;
            }
            if (el) {
                el.innerHTML = `
                    <div class="alert alert-success">
                        <strong>✅ ${data.message}</strong>
                        ${extra}
                    </div>
                    <button onclick="location.reload()" class="btn-outline-custom w-100 mt-2">
                        Refresh Page
                    </button>`;
            }
        } else {
            btn.disabled = false;
            btn.textContent = "Retry";
            showCameraError("Failed to save: " + (data.error || "Unknown error"));
        }
    })
    .catch(err => {
        btn.disabled = false;
        btn.textContent = "Retry";
        showCameraError("Network error: " + err.message);
    });
}

// Auto-start camera when page loads (only if video element exists)
document.addEventListener("DOMContentLoaded", function () {
    if (document.getElementById("videoIn"))  startCamera("videoIn");
    if (document.getElementById("videoOut")) startCamera("videoOut");
});