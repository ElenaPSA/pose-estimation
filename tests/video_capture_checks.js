function runVideoCaptureChecks(api) {
  if (api.sampleTimes(10).length !== 40) throw new Error("Expected four samples per second.");
  for (const duration of [.0001, .2, 1, 10, 120, 600, 3600]) {
    const times = api.sampleTimes(duration);
    if (!times.length || times.length > 240) throw new Error("Unbounded video sampling.");
    if (times.some((time, i) => time < 0 || time >= duration || (i > 0 && time <= times[i - 1]))) {
      throw new Error("Invalid sample times.");
    }
    if (Math.abs(times[0] + times[times.length - 1] - duration) > .0001) {
      throw new Error("Samples do not cover the full duration.");
    }
  }
  for (const duration of [NaN, Infinity, 0, -1, "10"]) {
    let failed = false;
    try { api.sampleTimes(duration); } catch { failed = true; }
    if (!failed) throw new Error("Invalid duration accepted.");
  }
  return ["Bounded sampling across the full duration", "Invalid durations rejected"];
}

function observeVideoDetectionSequence(sampledPreview = false) {
  const viewport = document.getElementById("detectionViewport");
  const image = document.getElementById("detectionImage");
  const video = document.getElementById("sourceVideo");
  const context = document.getElementById("detectionVideoFrame").getContext("2d");
  const originalDraw = context.drawImage;
  const mirroredTimes = [];
  context.drawImage = function(source, ...args) {
    if (source === video) mirroredTimes.push(video.currentTime);
    return originalDraw.call(this, source, ...args);
  };
  let scannedFrames = 0;
  const observer = new MutationObserver(records => {
    for (const record of records) {
      if (record.attributeName === "src" && image.getAttribute("src") &&
          !viewport.hidden && viewport.classList.contains("scanning")) {
        scannedFrames++;
      }
    }
  });
  observer.observe(image, {attributes: true, attributeFilter: ["src"]});
  return () => {
    observer.disconnect();
    context.drawImage = originalDraw;
    if (sampledPreview ? scannedFrames < 2 :
        mirroredTimes.length < 2 || mirroredTimes[mirroredTimes.length - 1] <= mirroredTimes[0]) {
      throw new Error("Video was not shown while scanning.");
    }
    if (!viewport.hidden || viewport.classList.contains("scanning")) {
      throw new Error("Scanning did not end after analysis.");
    }
    const figure = document.getElementById("morphologyFigure");
    const stage = document.getElementById("morphologyFigureStage");
    const frame = stage.querySelector("img").getBoundingClientRect();
    const overlay = stage.querySelector("svg").getBoundingClientRect();
    if (figure.hidden || frame.width <= 0 ||
        Math.abs(frame.width - overlay.width) > .5 || Math.abs(frame.height - overlay.height) > .5) {
      throw new Error("Morphology overlay is not aligned with the selected frame.");
    }
    const torso = document.getElementById("morphologyTorso");
    if (getComputedStyle(torso).display === "none" || !torso.hasAttribute("x1")) {
      throw new Error("Detected torso segment is not shown.");
    }
    if (!matchMedia("(prefers-reduced-motion: reduce)").matches &&
        getComputedStyle(torso).animationName !== "segment-reveal") {
      throw new Error("Detected segments do not reveal.");
    }
    return ["Playback during detection", "Aligned morphology overlay and segment reveal"];
  };
}

async function checkMirrorStopped() {
  const context = document.getElementById("detectionVideoFrame").getContext("2d");
  const originalDraw = context.drawImage;
  let draws = 0;
  context.drawImage = function(...args) {
    draws++;
    return originalDraw.apply(this, args);
  };
  try {
    await new Promise(resolve => setTimeout(resolve, 300));
    if (draws) throw new Error("Video mirror continued rendering after analysis.");
  } finally {
    context.drawImage = originalDraw;
  }
}
