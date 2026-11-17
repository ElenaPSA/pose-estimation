const WebcamCapture = (() => {
  function samePerson(previous, current) {
    if (!previous || !current) return false;
    const width = Math.max(0, Math.min(previous.x2, current.x2) - Math.max(previous.x1, current.x1));
    const height = Math.max(0, Math.min(previous.y2, current.y2) - Math.max(previous.y1, current.y1));
    const intersection = width * height;
    const previousArea = Math.max(0, previous.x2 - previous.x1) * Math.max(0, previous.y2 - previous.y1);
    const currentArea = Math.max(0, current.x2 - current.x1) * Math.max(0, current.y2 - current.y1);
    const union = previousArea + currentArea - intersection;
    const distance = Math.hypot(
      (previous.x1 + previous.x2 - current.x1 - current.x2) / 2,
      (previous.y1 + previous.y2 - current.y1 - current.y2) / 2
    );
    return (union > 0 && intersection / union >= .05) || distance <= .08 * Math.SQRT2;
  }

  function mount({captured, reset}) {
    const video = document.getElementById("sourceVideo");
    const startButton = document.getElementById("startCamera");
    const stopButton = document.getElementById("stopCamera");
    const retryButton = document.getElementById("retryCamera");
    const status = document.getElementById("profileStatus");
    const cameraStatus = document.getElementById("cameraStatus");
    const canvas = document.createElement("canvas");
    let stream = null;
    let controller = null;
    let generation = 0;
    let scanning = false;

    function stopCamera(message = "Camera stopped.") {
      generation++;
      controller?.abort();
      controller = null;
      stream?.getTracks().forEach(track => track.stop());
      stream = null;
      video.srcObject = null;
      video.hidden = true;
      scanning = false;
      startButton.disabled = false;
      stopButton.disabled = true;
      retryButton.disabled = true;
      cameraStatus.textContent = message;
      status.textContent = message;
    }

    async function scan(signal) {
      if (scanning) return;
      scanning = true;
      retryButton.disabled = true;
      reset();
      let previousBox = null;
      let observations = [];
      const startTime = performance.now();
      try {
        while (!signal.aborted && stream?.active) {
          if (!video.videoWidth || !video.videoHeight || video.readyState < 2) {
            await new Promise(resolve => setTimeout(resolve, 200));
            continue;
          }
          const scale = Math.min(1, 1280 / Math.max(video.videoWidth, video.videoHeight));
          canvas.width = Math.round(video.videoWidth * scale);
          canvas.height = Math.round(video.videoHeight * scale);
          const context = canvas.getContext("2d");
          if (!context) throw new Error("Camera frame capture is unavailable.");
          context.drawImage(video, 0, 0, canvas.width, canvas.height);
          const image = canvas.toDataURL("image/jpeg", .9);
          status.textContent = "Checking the live camera for a complete person…";
          const response = await fetch("/probe-frame", {
            method: "POST",
            signal: AbortSignal.any([signal, AbortSignal.timeout(90000)]),
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({mode: "webcam_demo", frontal: {image}})
          });
          const result = await response.json();
          if (!response.ok) throw new Error(result.error || "Full-body search failed.");
          signal.throwIfAborted();
          const candidate = result.eligible && result.confident && result.track_box;
          if (candidate) {
            if (previousBox && !samePerson(previousBox, result.track_box)) observations = [];
            previousBox = result.track_box;
            observations.push({image, timestamp: (performance.now() - startTime) / 1000, quality: Number(result.quality_score) || 0});
            if (observations.length > 3) observations.shift();
            if ((observations.length >= 2 && result.near_camera) || observations.length >= 3) {
              const best = observations.reduce((left, right) => right.quality > left.quality ? right : left);
              await captured(best.image, best.timestamp, signal);
              signal.throwIfAborted();
              cameraStatus.textContent = "Camera active. Capture complete.";
              return;
            }
            status.textContent = `Tracking the same person: ${observations.length}/3 checks.`;
          } else {
            previousBox = null;
            observations = [];
            status.textContent = result.reason || "Waiting for a high-confidence full-body detection.";
          }
          await new Promise(resolve => setTimeout(resolve, 250));
        }
      } catch (error) {
        if (!signal.aborted) {
          status.textContent = error.name === "TimeoutError" ? "Camera analysis timed out. Try again." : (error.message || "Camera analysis failed.");
          document.getElementById("detectionStatus").textContent = status.textContent;
        }
      } finally {
        if (controller?.signal === signal) {
          scanning = false;
          if (!signal.aborted && stream?.active) retryButton.disabled = false;
        }
      }
    }

    async function startCamera() {
      if (controller) return;
      const currentGeneration = ++generation;
      const currentController = new AbortController();
      controller = currentController;
      const {signal} = currentController;
      startButton.disabled = true;
      stopButton.disabled = false;
      retryButton.disabled = true;
      cameraStatus.textContent = "Requesting camera access…";
      status.textContent = "Requesting camera access…";
      try {
        if (!navigator.mediaDevices?.getUserMedia) throw new Error("Camera access requires localhost or HTTPS and a supported browser.");
        const newStream = await navigator.mediaDevices.getUserMedia({video: true, audio: false});
        if (signal.aborted || currentGeneration !== generation) {
          newStream.getTracks().forEach(track => track.stop());
          return;
        }
        stream = newStream;
        stream.getVideoTracks().forEach(track => track.addEventListener("ended", () => {
          if (currentGeneration === generation) stopCamera("Camera disconnected.");
        }, {once: true}));
        video.srcObject = stream;
        video.hidden = false;
        await video.play();
        signal.throwIfAborted();
        cameraStatus.textContent = "Camera active.";
        await scan(signal);
      } catch (error) {
        if (signal.aborted || currentGeneration !== generation) return;
        const message = error.name === "NotAllowedError" ? "Camera permission denied. Allow camera access and try again." :
          error.name === "NotFoundError" ? "No camera found. Connect a webcam and try again." :
          error.name === "NotReadableError" ? "Camera is unavailable or in use by another application." :
          (error.message || "Could not start the camera.");
        stopCamera(message);
      }
    }

    startButton.addEventListener("click", startCamera);
    stopButton.addEventListener("click", () => stopCamera());
    retryButton.addEventListener("click", () => {
      if (controller && stream?.active) void scan(controller.signal);
    });
    window.addEventListener("pagehide", () => stopCamera(), {once: true});
    void startCamera();
  }
  return {mount, samePerson};
})();