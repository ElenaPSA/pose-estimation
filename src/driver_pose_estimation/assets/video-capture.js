const VideoCapture = (() => {
  function sampleTimes(duration) {
    if (!Number.isFinite(duration) || duration <= 0) throw new Error("Video duration is unavailable.");
    const count = Math.min(240, Math.max(1, Math.ceil(duration * 2)));
    return Array.from({length: count}, (_, i) => (i + .5) * duration / count);
  }
  function waitFor(video, event, action, signal) {
    return new Promise((resolve, reject) => {
      const cleanup = () => {
        clearTimeout(timer);
        video.removeEventListener(event, done);
        video.removeEventListener("error", failed);
        signal.removeEventListener("abort", aborted);
      };
      const done = () => { cleanup(); resolve(); };
      const failed = () => { cleanup(); reject(new Error("Cannot decode this video. Try an MP4 (H.264) or WebM file.")); };
      const aborted = () => { cleanup(); reject(signal.reason); };
      const timer = setTimeout(() => {
        cleanup();
        reject(new Error("Video decoding timed out. Try another video."));
      }, 15000);
      video.addEventListener(event, done, {once: true});
      video.addEventListener("error", failed, {once: true});
      signal.addEventListener("abort", aborted, {once: true});
      if (signal.aborted) { aborted(); return; }
      action();
    });
  }
  function samePerson(previous, current) {
    if (!previous) return Boolean(current);
    if (!current) return false;
    const width = Math.max(0, Math.min(previous.x2, current.x2) - Math.max(previous.x1, current.x1));
    const height = Math.max(0, Math.min(previous.y2, current.y2) - Math.max(previous.y1, current.y1));
    const intersection = width * height;
    const previousArea = Math.max(0, previous.x2 - previous.x1) * Math.max(0, previous.y2 - previous.y1);
    const currentArea = Math.max(0, current.x2 - current.x1) * Math.max(0, current.y2 - current.y1);
    const union = previousArea + currentArea - intersection;
    const overlap = union > 0 ? intersection / union : 0;
    const previousX = (previous.x1 + previous.x2) / 2;
    const previousY = (previous.y1 + previous.y2) / 2;
    const currentX = (current.x1 + current.x2) / 2;
    const currentY = (current.y1 + current.y2) / 2;
    const distance = Math.hypot(previousX - currentX, previousY - currentY);
    return overlap >= .05 || distance <= .08 * Math.SQRT2;
  }
  function mount({captured, reset}) {
    const input = document.getElementById("videoFile");
    const video = document.getElementById("sourceVideo");
    const canvas = document.createElement("canvas");
    const status = document.getElementById("profileStatus");
    const preview = document.getElementById("decodedVideoPreview");
    let controller = null, url = null;
    window.addEventListener("pagehide", () => {
      controller?.abort();
      if (url) URL.revokeObjectURL(url);
    });
    input.addEventListener("change", async () => {
      const file = input.files[0];
      if (!file) return;
      controller = new AbortController();
      const {signal} = controller;
      input.disabled = true;
      reset();
      video.hidden = true;
      preview.hidden = true;
      try {
        if (file.size > 500 * 1024 * 1024 || !file.type.startsWith("video/")) {
          throw new Error("Choose a browser-playable video smaller than 500 MB.");
        }
        video.pause();
        if (url) URL.revokeObjectURL(url);
        url = URL.createObjectURL(file);
        status.textContent = "Loading video locally…";
        try {
          await waitFor(video, "loadeddata", () => { video.src = url; video.load(); }, signal);
        } catch (error) {
          if (signal.aborted) throw error;
          status.textContent = "Browser cannot play this codec. Starting local decoded preview…";
          video.hidden = true;
          preview.hidden = false;
          const response = await fetch("/stream-video", {
            method: "POST",
            signal: AbortSignal.any([signal, AbortSignal.timeout(600000)]),
            headers: {"Content-Type": "application/octet-stream"},
            body: file
          });
          if (!response.ok) {
            const result = await response.json();
            throw new Error(result.error || "Local video playback failed.");
          }
          if (!response.body) throw new Error("Streaming video preview is unavailable.");
          const reader = response.body.getReader();
          const decoder = new TextDecoder();
          let buffer = "";
          const consume = async line => {
            if (!line.trim()) return false;
            const event = JSON.parse(line);
            signal.throwIfAborted();
            if (event.error) throw new Error(event.error);
            if (event.type === "frame") {
              preview.src = event.image;
              status.textContent = `Local decoded preview (sampled): ${event.timestamp.toFixed(2)} s. Checking full-body visibility…`;
              await new Promise(resolve => setTimeout(resolve, 500));
              signal.throwIfAborted();
            } else if (event.type === "captured") {
              preview.src = event.image;
              await captured(event.image, event.timestamp, signal);
              return true;
            }
            return false;
          };
          try {
            while (true) {
              const {value, done} = await reader.read();
              buffer += decoder.decode(value, {stream: !done});
              let end;
              while ((end = buffer.indexOf("\n")) >= 0) {
                const line = buffer.slice(0, end);
                buffer = buffer.slice(end + 1);
                if (await consume(line)) return;
              }
              if (done) {
                if (await consume(buffer)) return;
                throw new Error("Video preview ended without a confirmed full-body frame.");
              }
            }
          } finally {
            await reader.cancel();
          }
        }
        video.hidden = false;
        sampleTimes(video.duration);
        const scale = Math.min(1, 1280 / Math.max(video.videoWidth, video.videoHeight));
        canvas.width = Math.round(video.videoWidth * scale);
        canvas.height = Math.round(video.videoHeight * scale);
        const context = canvas.getContext("2d");
        if (!context || !canvas.width || !canvas.height) throw new Error("Video frame capture is unavailable.");
        await video.play();
        let confirmations = 0;
        let examined = 0;
        let trackedBox = null;
        let recentObservations = [];
        let bestObservation = null;
        let bestQuality = -1;
        let reason = "No suitable person detected.";
        while (!video.ended) {
          signal.throwIfAborted();
          const time = video.currentTime;
          status.textContent = `Playing video / checking full body at ${time.toFixed(2)} s. ${reason}`;
          context.drawImage(video, 0, 0, canvas.width, canvas.height);
          const image = canvas.toDataURL("image/jpeg", .9);
          const response = await fetch("/probe-frame", {
            method: "POST",
            signal: AbortSignal.any([signal, AbortSignal.timeout(90000)]),
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({frontal: {image}})
          });
          const result = await response.json();
          if (!response.ok) throw new Error(result.error || "Full-body search failed.");
          signal.throwIfAborted();
          examined++;
          const currentBox = result.track_box;
          const validCandidate = result.eligible && result.confident && currentBox;
          const sameTrack = trackedBox
            ? samePerson(trackedBox, currentBox)
            : Boolean(validCandidate);
          if (sameTrack) trackedBox = currentBox;
          const confident = Boolean(validCandidate && sameTrack);
          if (confident) {
            const observation = {
              image,
              timestamp: time,
              quality: Number(result.quality_score) || 0,
              relativeHeight: Number(result.relative_height) || 0,
              nearCamera: Boolean(result.near_camera)
            };
            recentObservations.push(observation);
            if (recentObservations.length > 3) recentObservations.shift();
            confirmations = recentObservations.length;
            if (confirmations >= 2 && observation.nearCamera) {
              const selectedObservation = recentObservations.reduce(
                (best, candidate) => candidate.quality > best.quality ? candidate : best
              );
              video.pause();
              if (Math.abs(video.currentTime - selectedObservation.timestamp) > .001) {
                await waitFor(video, "seeked", () => {
                  video.currentTime = selectedObservation.timestamp;
                }, signal);
              }
              status.textContent = `Complete near-camera frame found. Scanning at ${selectedObservation.timestamp.toFixed(2)} s…`;
              await captured(selectedObservation.image, selectedObservation.timestamp, signal);
              return;
            }
            if (confirmations >= 3) {
              for (const candidate of recentObservations) {
                if (candidate.quality > bestQuality) {
                  bestObservation = candidate;
                  bestQuality = candidate.quality;
                }
              }
            }
            reason = observation.nearCamera
              ? `Near-camera frame confirmed ${confirmations}/3 times.`
              : `Tracking the same complete body: ${confirmations}/3 consecutive frames.`;
          } else {
            recentObservations = [];
            confirmations = 0;
            reason = result.eligible && sameTrack
              ? "Detection confidence is below the required level."
              : result.reason || "The detected person changed or was not confirmed.";
          }
          status.textContent = `Playing video / checking ${time.toFixed(2)} s of ${video.duration.toFixed(2)} s. ${reason}`;
          if (video.ended) break;
          await new Promise(resolve => setTimeout(resolve, 500));
          if (video.ended) break;
        }
        if (!bestObservation) {
          throw new Error(`No high-confidence complete-body frame in ${examined} checks. ${reason} Move closer while keeping the head and both feet inside the frame.`);
        }
        status.textContent = `Video ended. Scanning the closest complete frame at ${bestObservation.timestamp.toFixed(2)} s…`;
        await captured(bestObservation.image, bestObservation.timestamp, signal);
      } catch (error) {
        status.textContent = signal.aborted ? "Video analysis cancelled. Choose a video to try again." :
          error.name === "TimeoutError" ? "Local inference timed out. Check the server/model setup and try again." :
          error.message || "Video processing failed.";
        document.getElementById("detectionStatus").textContent = status.textContent;
      } finally {
        video.pause();
        input.disabled = false;
        input.value = "";
        controller = null;
      }
    });
  }
  return {mount, sampleTimes};
})();
