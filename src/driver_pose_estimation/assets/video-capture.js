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
  function mount({captured, reset}) {
    const input = document.getElementById("videoFile");
    const video = document.getElementById("sourceVideo");
    const canvas = document.createElement("canvas");
    const status = document.getElementById("profileStatus");
    const cancel = document.getElementById("cancelVideo");
    const preview = document.getElementById("decodedVideoPreview");
    let controller = null, url = null;
    cancel.addEventListener("click", () => controller?.abort());
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
      cancel.disabled = false;
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
        let reason = "No suitable person detected.";
        while (examined < 240) {
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
          confirmations = result.eligible ? confirmations + 1 : 0;
          if (confirmations >= 2) {
            video.pause();
            if (Math.abs(video.currentTime - time) > .001) {
              await waitFor(video, "seeked", () => { video.currentTime = time; }, signal);
            }
            await captured(image, time, signal);
            return;
          }
          reason = result.eligible ? "Full body found; confirming in another frame…" : result.reason;
          if (video.ended) break;
          await new Promise(resolve => setTimeout(resolve, 500));
          if (video.ended) break;
        }
        throw new Error(`No confirmed full-body frame in ${examined} checks. ${reason} Use one upright, front-facing person with head and feet inside the frame and a clear face.`);
      } catch (error) {
        status.textContent = signal.aborted ? "Video analysis cancelled. Choose a video to try again." :
          error.name === "TimeoutError" ? "Local inference timed out. Check the server/model setup and try again." :
          error.message || "Video processing failed.";
        document.getElementById("detectionStatus").textContent = status.textContent;
      } finally {
        video.pause();
        input.disabled = false;
        input.value = "";
        cancel.disabled = true;
        controller = null;
      }
    });
  }
  return {mount, sampleTimes};
})();
