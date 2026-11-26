const VideoCapture = (() => {
  function sampleTimes(duration) {
    if (!Number.isFinite(duration) || duration <= 0) throw new Error("Video duration is unavailable.");
    const count = Math.min(240, Math.max(1, Math.ceil(duration * 4)));
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
  function mount({captured, frozen, reset, progress, playback, failed}) {
    const input = document.getElementById("videoFile");
    const fileStatus = document.getElementById("videoFileStatus");
    const video = document.getElementById("sourceVideo");
    const scanVideo = document.createElement("video");
    scanVideo.muted = true;
    scanVideo.playsInline = true;
    scanVideo.preload = "auto";
    const canvas = document.createElement("canvas");
    const status = document.getElementById("profileStatus");
    const preview = document.getElementById("decodedVideoPreview");
    let controller = null, url = null;
    const liveControls = document.getElementById("liveCameraControls");
    const liveButton = document.getElementById("liveCameraButton");
    const liveStatus = document.getElementById("liveCameraStatus");
    let liveController = null;
    if (liveControls && liveButton) {
      fetch("/camera-status", {cache: "no-store"})
        .then(response => response.ok ? response.json() : null)
        .then(camera => {
          if (!camera?.enabled) return;
          liveButton.textContent = `Use live camera (UDP ${camera.port})`;
          liveStatus.textContent = camera.receiving
            ? "Camera stream received."
            : `Waiting for camera stream on UDP port ${camera.port}.`;
          liveControls.hidden = false;
          liveButton.addEventListener("click", () => {
            if (liveController) liveController.abort();
            else scanLiveCamera(camera.port);
          });
        })
        .catch(() => {});
    }
    window.addEventListener("pagehide", () => {
      liveController?.abort();
      controller?.abort();
      if (url) URL.revokeObjectURL(url);
    });
    async function load(file) {
      if (!file || controller) return;
      fileStatus.textContent = `Selected video: ${file.name}`;
      controller = new AbortController();
      const {signal} = controller;
      input.disabled = true;
      if (liveButton) liveButton.disabled = true;
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
          try {
            status.textContent = "Converting the local video to browser-compatible MP4…";
            const response = await fetch("/transcode-video", {
              method: "POST",
              signal: AbortSignal.any([signal, AbortSignal.timeout(600000)]),
              headers: {"Content-Type": "application/octet-stream"},
              body: file
            });
            if (!response.ok) {
              const result = await response.json();
              throw new Error(result.error || "Browser-compatible transcode failed.");
            }
            const previewUrl = URL.createObjectURL(await response.blob());
            await waitFor(video, "loadeddata", () => {
              video.src = previewUrl;
              video.load();
            }, signal);
            if (url) URL.revokeObjectURL(url);
            url = previewUrl;
            status.textContent = "Playing converted local video while checking the full body…";
          } catch (transcodeError) {
            if (signal.aborted) throw transcodeError;
            status.textContent = "Browser transcode unavailable. Starting sampled local preview…";
            video.hidden = true;
            preview.hidden = false;
            const streamResponse = await fetch("/stream-video", {
              method: "POST",
              signal: AbortSignal.any([signal, AbortSignal.timeout(600000)]),
              headers: {"Content-Type": "application/octet-stream"},
              body: file
            });
            if (!streamResponse.ok) {
              const result = await streamResponse.json();
              throw new Error(result.error || "Local video preview failed.");
            }
            if (!streamResponse.body) throw new Error("Streaming video preview is unavailable.");
            const reader = streamResponse.body.getReader();
            const decoder = new TextDecoder();
            let buffer = "";
            let frozenPreview = false;
            const consume = async line => {
              if (!line.trim()) return false;
              const event = JSON.parse(line);
              signal.throwIfAborted();
              if (event.error) throw new Error(event.error);
              if (event.type === "frame") {
                preview.src = event.image;
                if (!frozenPreview) progress(event.image, event.timestamp);
                status.textContent = `Local decoded preview (sampled frames): ${event.timestamp.toFixed(2)} s. Checking full-body visibility…`;
                signal.throwIfAborted();
              } else if (event.type === "frozen") {
                frozenPreview = true;
                await frozen(event.image, event.timestamp, signal);
              } else if (event.type === "captured") {
                preview.src = event.image;
                await captured(event.image, event.timestamp, signal, true, event.height_samples_cm || []);
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
        }
        video.hidden = false;
        const times = sampleTimes(video.duration);
        await waitFor(scanVideo, "loadeddata", () => {
          scanVideo.src = url;
          scanVideo.load();
        }, signal);
        const scale = Math.min(1, 1280 / Math.max(video.videoWidth, video.videoHeight));
        canvas.width = Math.round(video.videoWidth * scale);
        canvas.height = Math.round(video.videoHeight * scale);
        const context = canvas.getContext("2d");
        if (!context || !canvas.width || !canvas.height) throw new Error("Video frame capture is unavailable.");
        await video.play();
        playback(video);
        let confirmations = 0;
        let examined = 0;
        let trackedBox = null;
        let recentObservations = [];
        let bestObservation = null;
        let bestQuality = -1;
        let reason = "No suitable person detected.";
        let frozenObservation = null;
        const heightSamples = [];
        const probe = async (source, time, nearby = false) => {
          context.drawImage(source, 0, 0, canvas.width, canvas.height);
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
          const referenceBox = nearby ? frozenObservation.trackBox : trackedBox;
          const sameTrack = referenceBox
            ? samePerson(referenceBox, currentBox)
            : Boolean(validCandidate);
          if (nearby && !sameTrack) return false;
          if (sameTrack) trackedBox = currentBox;
          if (validCandidate && sameTrack) {
            const observation = {
              image,
              timestamp: time,
              quality: Number(result.quality_score) || 0,
              relativeHeight: Number(result.relative_height) || 0,
              nearCamera: Boolean(result.near_camera),
              height: result.height_cm,
              trackBox: currentBox
            };
            recentObservations.push(observation);
            if (recentObservations.length > 3) recentObservations.shift();
            confirmations = recentObservations.length;
            if ((nearby || confirmations >= 2) && observation.nearCamera &&
                Number.isFinite(observation.height)) {
              if (!frozenObservation) {
                frozenObservation = observation;
                await frozen(observation.image, observation.timestamp, signal);
              }
              heightSamples.push(observation.height);
              reason = `Collecting close-up height measurements: ${heightSamples.length}/5.`;
              if (!nearby || heightSamples.length >= 5) return true;
            }
            if (confirmations >= 3) {
              for (const candidate of recentObservations) {
                if (candidate.quality > bestQuality) {
                  bestObservation = candidate;
                  bestQuality = candidate.quality;
                }
              }
            }
            if (!frozenObservation) {
              reason = observation.nearCamera
                ? `Near-camera frame confirmed ${confirmations}/3 times.`
                : `Tracking the same complete body: ${confirmations}/3 consecutive frames.`;
            }
          } else {
            recentObservations = [];
            confirmations = 0;
            reason = result.eligible && sameTrack
              ? "Detection confidence is below the required level."
              : result.reason || "The detected person changed or was not confirmed.";
          }
          status.textContent = `Scanning video at ${time.toFixed(2)} s of ${video.duration.toFixed(2)} s. ${reason}`;
          return false;
        };
        const finish = async observation => {
          if (heightSamples.length) {
            for (const offset of [-.2, .2, -.4, .4, -.6, .6, -.8, .8]) {
              const time = observation.timestamp + offset;
              if (time <= 0 || time >= video.duration) continue;
              signal.throwIfAborted();
              await waitFor(scanVideo, "seeked", () => { scanVideo.currentTime = time; }, signal);
              if (await probe(scanVideo, time, true)) break;
            }
          }
          status.textContent = `Analyzing the frozen frame and ${heightSamples.length} close-up height measurements…`;
          await captured(observation.image, observation.timestamp, signal, false, heightSamples);
        };
        // Probe the frames being played so detection stays in step with both figures.
        let lastProbeTime = -1;
        while (!video.ended) {
          signal.throwIfAborted();
          const time = video.currentTime;
          if (video.paused || time === lastProbeTime || video.readyState < 2) {
            await new Promise(resolve => setTimeout(resolve, 80));
            continue;
          }
          lastProbeTime = time;
          const complete = await probe(video, time);
          if (complete) {
            await finish(frozenObservation);
            return;
          }
        }
        if (frozenObservation) {
          await finish(frozenObservation);
          return;
        }
        if (!bestObservation) {
          recentObservations = [];
          confirmations = 0;
          trackedBox = null;
          for (const time of times) {
            signal.throwIfAborted();
            await waitFor(scanVideo, "seeked", () => { scanVideo.currentTime = time; }, signal);
            const complete = await probe(scanVideo, time);
            if (complete) {
              await finish(frozenObservation);
              return;
            }
          }
        }
        if (frozenObservation) {
          await finish(frozenObservation);
          return;
        }
        if (!bestObservation) {
          throw new Error(`No high-confidence complete-body frame in ${examined} checks. ${reason} Move closer while keeping the head and both feet inside the frame.`);
        }
        status.textContent = `Video scan complete. Analyzing the closest complete frame at ${bestObservation.timestamp.toFixed(2)} s…`;
        await captured(bestObservation.image, bestObservation.timestamp, signal);
      } catch (error) {
        failed();
        status.textContent = signal.aborted ? "Video analysis cancelled. Choose a video to try again." :
          error.name === "TimeoutError" ? "Local inference timed out. Check the server/model setup and try again." :
          error.message || "Video processing failed.";
        document.getElementById("detectionStatus").textContent = status.textContent;
      } finally {
        if (signal.aborted) {
          video.pause();
        }
        scanVideo.removeAttribute("src");
        scanVideo.load();
        input.disabled = false;
        if (liveButton) liveButton.disabled = false;
        input.value = "";
        controller = null;
      }
    }
    input.addEventListener("change", () => load(input.files[0]));
    async function scanLiveCamera(port) {
      if (controller) return;
      liveController = new AbortController();
      const {signal} = liveController;
      const label = liveButton.textContent;
      liveButton.textContent = "Stop live camera";
      input.disabled = true;
      reset();
      video.pause();
      video.hidden = true;
      preview.hidden = false;
      const started = performance.now();
      let lastFrameId = null;
      let confirmations = 0;
      let trackedBox = null;
      let recentObservations = [];
      let frozenObservation = null;
      const heightSamples = [];
      let reason = "No suitable person detected.";
      try {
        while (true) {
          signal.throwIfAborted();
          const frameResponse = await fetch("/camera-frame", {signal, cache: "no-store"});
          const frameId = frameResponse.headers.get("X-Frame-Id");
          if (!frameResponse.ok || frameId === lastFrameId) {
            await frameResponse.body?.cancel();
            if (!frameResponse.ok) {
              liveStatus.textContent = `Waiting for camera stream on UDP port ${port}…`;
            }
            await new Promise(resolve => setTimeout(resolve, 100));
            continue;
          }
          lastFrameId = frameId;
          liveStatus.textContent = "Camera stream received.";
          const blob = await frameResponse.blob();
          const image = await new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.onerror = () => reject(new Error("Could not read the camera frame."));
            reader.readAsDataURL(blob);
          });
          const timestamp = (performance.now() - started) / 1000;
          preview.src = image;
          if (!frozenObservation) progress(image, timestamp);
          status.textContent = `Live camera at ${timestamp.toFixed(1)} s. ${reason}`;
          const response = await fetch("/probe-frame", {
            method: "POST",
            signal: AbortSignal.any([signal, AbortSignal.timeout(90000)]),
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({frontal: {image}})
          });
          const result = await response.json();
          if (!response.ok) throw new Error(result.error || "Full-body search failed.");
          signal.throwIfAborted();
          const currentBox = result.track_box;
          const liveCandidate = result.eligible && result.confident && currentBox;
          const sameTrack = trackedBox ? samePerson(trackedBox, currentBox) : Boolean(liveCandidate);
          if (sameTrack) trackedBox = currentBox;
          if (liveCandidate && sameTrack) {
            const observation = {
              image,
              timestamp,
              quality: Number(result.quality_score) || 0,
              nearCamera: Boolean(result.near_camera),
              height: result.height_cm
            };
            recentObservations.push(observation);
            if (recentObservations.length > 3) recentObservations.shift();
            confirmations = recentObservations.length;
            if ((frozenObservation || confirmations >= 2) && observation.nearCamera &&
                Number.isFinite(observation.height)) {
              if (!frozenObservation) {
                frozenObservation = recentObservations.reduce(
                  (best, candidate) => candidate.quality > best.quality ? candidate : best
                );
                await frozen(frozenObservation.image, frozenObservation.timestamp, signal);
              }
              heightSamples.push(observation.height);
              if (heightSamples.length >= 5) {
                status.textContent = `Analyzing the frozen frame and ${heightSamples.length} close-up height measurements…`;
                await captured(frozenObservation.image, frozenObservation.timestamp, signal, false, heightSamples);
                return;
              }
              reason = `Collecting close-up height measurements: ${heightSamples.length}/5.`;
            } else if (!frozenObservation) {
              reason = observation.nearCamera
                ? `Near-camera frame confirmed ${confirmations}/2 times.`
                : "Complete body tracked. Move closer to the camera.";
            }
          } else if (!frozenObservation) {
            recentObservations = [];
            confirmations = 0;
            reason = result.eligible && sameTrack
              ? "Detection confidence is below the required level."
              : result.reason || "The detected person changed or was not confirmed.";
          }
          status.textContent = `Live camera at ${timestamp.toFixed(1)} s. ${reason}`;
        }
      } catch (error) {
        failed();
        status.textContent = signal.aborted ? "Live camera analysis stopped." :
          error.name === "TimeoutError" ? "Local inference timed out. Check the server/model setup and try again." :
          error.message || "Live camera processing failed.";
        document.getElementById("detectionStatus").textContent = status.textContent;
      } finally {
        liveButton.textContent = label;
        input.disabled = false;
        liveController = null;
      }
    }
    return {load};
  }
  return {mount, sampleTimes};
})();
