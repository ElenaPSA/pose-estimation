async function checkSeatPlaybackCompletion(api) {
  api.update(178, 50);
  const interrupted = api.play();
  api.stop();
  if (await interrupted !== false) throw new Error("Interrupted animation reported completion.");
  if (await api.play() !== true) throw new Error("Animation did not report completion.");
  if (document.getElementById("studioProgress").style.width !== "100%") {
    throw new Error("Animation completed before the final frame.");
  }
  return ["Animation cancellation", "Animation completion after final frame"];
}

async function checkHmiStepSequences({analyze, advance, finishAnimation}) {
  const activeScreen = () => document.querySelector('[data-screen][aria-pressed="true"]').dataset.screen;
  const expectScreen = screen => {
    if (activeScreen() !== screen) throw new Error(`Expected ${screen}, got ${activeScreen()}.`);
  };
  const first = await analyze();
  expectScreen("capture");
  await advance(5999);
  expectScreen("capture");
  await advance(1);
  expectScreen("adjustments");
  await advance(5999);
  expectScreen("adjustments");
  await advance(1);
  expectScreen("animation");
  await advance(6000);
  expectScreen("animation");
  await finishAnimation();
  await advance(5999);
  expectScreen("animation");
  await advance(1);
  expectScreen("reference");
  await first.completion;

  const second = await analyze();
  expectScreen("capture");
  await advance(12000);
  expectScreen("animation");
  const replacement = await analyze();
  await second.completion;
  expectScreen("capture");
  await advance(6000);
  expectScreen("adjustments");
  document.querySelector('[data-screen="reference"]').click();
  await replacement.completion;
  await advance(6000);
  expectScreen("reference");

  const afterCancellation = await analyze();
  expectScreen("capture");
  await advance(6000);
  expectScreen("adjustments");
  document.querySelector('[data-screen="capture"]').click();
  await afterCancellation.completion;
  return [
    "Six-second pauses and animation-completion wait",
    "Every successful analysis restarts the sequence",
    "Replacement analysis ignores stale animation cancellation",
    "Manual cancellation does not disable the next analysis"
  ];
}

function checkCaptureLayout() {
  const video = document.getElementById("sourceVideo");
  const frame = document.getElementById("morphologyFigureStage");
  if (video.hidden || document.getElementById("morphologyFigure").hidden) {
    throw new Error("Load and analyze a video before checking the capture layout.");
  }
  const videoRect = video.getBoundingClientRect();
  const frameRect = frame.getBoundingClientRect();
  const profile = document.querySelector(".profile-card").getBoundingClientRect();
  const capture = document.getElementById("captureScreen").getBoundingClientRect();
  const main = document.querySelector("main").getBoundingClientRect();
  if (profile.top < Math.max(videoRect.bottom, frameRect.bottom)) {
    throw new Error("Profile controls must be below both figures.");
  }
  if (capture.width < main.width * .98 || frameRect.width <= 0 || frameRect.height <= 0) {
    throw new Error("Capture panels must use the available width and show the selected frame.");
  }
  if (videoRect.height < Math.min(innerHeight * .4, 650)) throw new Error("Video preview is too small.");
  if (innerWidth >= 900 && Math.max(videoRect.bottom, frameRect.bottom) > innerHeight) {
    throw new Error("Both figures should fit vertically in a desktop window.");
  }
  if (document.documentElement.scrollWidth > innerWidth + 1) {
    throw new Error("Capture layout overflows the viewport horizontally.");
  }
  const heights = [...document.querySelectorAll("[data-size] .height-person")]
    .map(body => body.getBoundingClientRect().height);
  if (heights.length !== 3 || !(heights[0] < heights[1] && heights[1] < heights[2])) {
    throw new Error("Size illustrations must show increasing human stature.");
  }
  const widths = [...document.querySelectorAll("[data-build] .build-torso")]
    .map(body => body.getBoundingClientRect().width);
  if (widths.length !== 3 || !(widths[1] < widths[0] && widths[1] < widths[2])) {
    throw new Error("Build illustrations must show visibly different body proportions.");
  }
  if (getComputedStyle(document.querySelector(".size-icon")).fill ===
      getComputedStyle(document.querySelector(".build-icon")).fill) {
    throw new Error("Height and build illustrations must use distinct visual styles.");
  }
  for (const button of document.querySelectorAll("[data-size], [data-build]")) {
    const icon = button.querySelector("svg");
    if (!icon || icon.getAttribute("aria-hidden") !== "true" ||
        icon.getAttribute("focusable") !== "false") {
      throw new Error("Human illustrations must preserve the button's text-only accessible name.");
    }
    if (icon.getBoundingClientRect().height < 100 || !icon.querySelector(".mannequin")) {
      throw new Error("Profile illustrations must be large enough to show their technical mannequins.");
    }
    if (icon.querySelector(".figure-skin, .figure-hair, .figure-trousers, .build-contour")) {
      throw new Error("Profile mannequins must not retain character styling or muscle contours.");
    }
    const template = icon.querySelector("image");
    if (!template || !template.getAttribute("href").startsWith("/assets/profile-mannequin")) {
      throw new Error("Profile figures must use the supplied mannequin template.");
    }
  }
  if (innerWidth > 700) {
    const buttons = [...document.querySelectorAll("[data-size], [data-build]")];
    const first = buttons[0];
    const card = first.getBoundingClientRect();
    const figure = first.querySelector("svg").getBoundingClientRect();
    const baseline = first.querySelector("image").getBoundingClientRect().bottom;
    for (const button of buttons) {
      if (Math.abs(button.getBoundingClientRect().height - card.height) > 1 ||
          Math.abs(button.querySelector("svg").getBoundingClientRect().top - figure.top) > 1 ||
          Math.abs(button.querySelector("image").getBoundingClientRect().bottom - baseline) > 1) {
        throw new Error("Size and build figures must share aligned cards and foot baselines.");
      }
    }
  }
  return ["Full-width figures", "Controls below figures", "Larger preview", "Distinct height and build illustrations", "Decorative human icons"];
}

function checkSeatAdjustmentLayout() {
  if (document.getElementById("seatStudio").dataset.activeScreen !== "adjustments") {
    throw new Error("Open Seat adjustments before checking its layout.");
  }
  const cards = [...document.querySelectorAll(".movement-row")];
  if (cards.length !== 3) throw new Error("All three adjustment cards must remain available.");
  for (const card of cards) {
    const bounds = card.getBoundingClientRect();
    const drawing = card.querySelector("svg").getBoundingClientRect();
    const seat = card.querySelector(".studio-solid").getBoundingClientRect();
    if (seat.height < drawing.width * .4) throw new Error("Seat drawing is too small for its card.");
    if (innerWidth >= 900 && innerHeight >= 620 &&
        (bounds.height < innerHeight * .6 || innerHeight - bounds.bottom > 120)) {
      throw new Error("Adjustment cards must use the available vertical space.");
    }
    for (const element of card.querySelectorAll(".studio-solid, .studio-ghost, .studio-steering-wheel")) {
      const figure = element.getBoundingClientRect();
      if (figure.left < drawing.left || figure.right > drawing.right ||
          figure.top < drawing.top || figure.bottom > drawing.bottom) {
        throw new Error("Enlarged drawings must retain the whole seat and steering wheel.");
      }
    }
  }
  if (document.documentElement.scrollWidth > innerWidth + 1) {
    throw new Error("Adjustment cards overflow the window horizontally.");
  }
  return ["Taller adjustment cards", "Larger seats", "Uncropped seats and wheels", "No horizontal overflow"];
}

function checkReferenceLayout() {
  if (document.getElementById("seatStudio").dataset.activeScreen !== "reference") {
    throw new Error("Open Reference details before checking its layout.");
  }
  const schema = document.getElementById("studioReferenceSchema");
  const diagram = schema.getBoundingClientRect();
  const container = schema.parentElement.getBoundingClientRect();
  if (innerWidth > 900) {
    if (diagram.width < document.querySelector("main").getBoundingClientRect().width * .6 ||
        (innerHeight >= 620 && diagram.height < innerHeight * .6) ||
        document.getElementById("reference-nominal").getBoundingClientRect().height <
          Math.min(280, diagram.width * .38)) {
      throw new Error("Reference diagram must use the available window space.");
    }
  } else if (document.querySelector(".studio-details > div:nth-child(2)")
      .getBoundingClientRect().top < container.bottom) {
    throw new Error("Reference explanations must move below the diagram in narrow windows.");
  }
  for (const element of schema.querySelectorAll("#reference-nominal, .studio-steering-wheel, text")) {
    const bounds = element.getBoundingClientRect();
    if (bounds.left < diagram.left || bounds.right > diagram.right ||
        bounds.top < diagram.top || bounds.bottom > diagram.bottom) {
      throw new Error("Reference seat, steering wheel and measurement labels must remain uncropped.");
    }
  }
  if (document.documentElement.scrollWidth > innerWidth + 1) {
    throw new Error("Reference details overflow the window horizontally.");
  }
  return ["Expanded reference diagram", "Uncropped seat and measurement labels", "Responsive explanations"];
}

function runSeatStudioChecks(api) {
  const checks = [];
  function check(name, callback) { callback(); checks.push(name); }
  function equal(value, expected) {
    if (JSON.stringify(value) !== JSON.stringify(expected)) {
      throw new Error(`Expected ${JSON.stringify(expected)}, got ${JSON.stringify(value)}`);
    }
  }
  check("Two reported trials anchor a full-range linear model", () => {
    equal(api.compute(163), {x: 177.5, z: 30, recline: null});
    equal(api.compute(174), {x: 127.5, z: 40, recline: null});
    equal(api.compute(185), {x: 77.5, z: 50, recline: null});
    equal(api.fittingRange, [163, 185]);
    if (!api.supports(80) || !api.supports(260)) throw new Error("Model does not cover supported height range.");
    equal(api.compute(80), {x: 255, z: 0, recline: null});
    equal(api.compute(260), {x: 0, z: 80, recline: null});
  });
  check("Extrapolation is monotonic and clipped to travel limits", () => {
    let previous = api.compute(80);
    for (let height = 80.25; height <= 260; height += .25) {
      const current = api.compute(height);
      if (current.x > previous.x || current.z < previous.z) throw new Error(`Wrong adjustment direction at ${height} cm.`);
      for (const axis of ["x", "z"]) {
        if (current[axis] < api.limits[axis][0] || current[axis] > api.limits[axis][1]) throw new Error(`Out of bounds: ${height}, ${axis}`);
      }
      previous = current;
    }
  });
  check("Illustrative backrest uses torso geometry with a bounded adjustment", () => {
    const nominal = api.compute(174, 50).recline;
    if (Math.abs(nominal - 25) > 1e-10) throw new Error("Nominal torso must match the reference angle.");
    const longerTorso = api.compute(174, 55).recline;
    const shorterTorso = api.compute(174, 45).recline;
    if (!(longerTorso < 25 && shorterTorso > 25)) throw new Error("Unexpected torso-to-angle direction.");
    if (Math.abs(55 * Math.sin(longerTorso * Math.PI / 180) - 50 * Math.sin(25 * Math.PI / 180)) > 1e-10) {
      throw new Error("Shoulder offset is not preserved.");
    }
    equal(api.compute(174, 30).recline, 30);
    equal(api.compute(174, 85).recline, 20);
    for (const torso of [null, undefined, 0, -1, 25, 88, NaN, Infinity, "50"]) {
      equal(api.compute(174, torso).recline, null);
    }
  });
  check("Invalid heights are errors, never default targets", () => {
    for (const value of [79, 261, NaN, Infinity, "178", null, undefined]) {
      let failed = false;
      try { api.compute(value); } catch { failed = true; }
      if (!failed) throw new Error(`Accepted invalid input ${value}`);
    }
  });
  check("Only unvalidated extrapolation displays a fit warning", () => {
    api.update(174);
    if (!document.getElementById("seatFitModelStatus").hidden ||
        document.getElementById("seatFitModelStatus").textContent) {
      throw new Error("Interpolation message should not be displayed.");
    }
    equal(api.compute(174).recline, null);
    api.update(190);
    if (document.getElementById("seatFitModelStatus").hidden ||
        !document.getElementById("seatFitModelStatus").textContent.includes("extrapolated")) {
      throw new Error("Missing extrapolation warning.");
    }
    equal(api.compute(190).recline, null);
  });
  check("Positive X moves right, positive Z up, recline rearward left", () => {
    const low = api.transform({x: 0, z: 0, recline: 15});
    const high = api.transform({x: 255, z: 80, recline: 55});
    if (high.x <= low.x || high.y >= low.y || high.angle >= low.angle) {
      throw new Error("Screen orientation disagrees with vehicle coordinates.");
    }
    equal(high.x - low.x, 191.25);
    equal(low.y - high.y, 60);
    equal(api.transform(api.compute(178)).angle, -25);
  });
  check("Movements are target minus the nominal comfort reference", () => {
    equal(api.reference, {x: 127.5, z: 40, recline: 25});
    equal(api.delta(api.compute(163)), {x: 50, z: -10, recline: null});
    equal(api.delta(api.compute(185)), {x: -50, z: 10, recline: null});
    equal(api.delta(api.reference), {x: 0, z: 0, recline: 0});
    equal(api.delta({x: 0, z: 0, recline: 15}, {x:255, z:80, recline:55}),
      {x: -255, z: -80, recline: -40});
  });
  check("Cards show signed movements and fixed overlaid reference seats", () => {
    api.update(185, 55);
    for (const [axis, id, expected] of [["x", "seatDepth", "-50"], ["z", "seatHeight", "+10"], ["recline", "backrest", "-2.4"]]) {
      equal(document.getElementById(id).textContent, expected);
      const card = document.getElementById(`studio-card-${axis}`);
      const ghost = card.querySelector(".studio-ghost"), solid = card.querySelector(".studio-solid");
      const t = api.transform(api.reference);
      equal(ghost.getAttribute("transform"), `translate(${t.x} ${t.y})`);
      equal(ghost.querySelector(".studio-backrest").getAttribute("transform"), `rotate(${t.angle} 0 -18)`);
      if (!(ghost.compareDocumentPosition(solid) & Node.DOCUMENT_POSITION_FOLLOWING)) {
        throw new Error("Orange simulated seat must be drawn over the gray reference.");
      }
      if (axis !== "recline" && !document.getElementById(`studio-target-${axis}`).textContent.includes("Reference")) {
        throw new Error("Missing absolute reference-to-target caption.");
      }
    }
    equal(document.getElementById("studioLiveX").textContent, "-50");
    equal(document.getElementById("studioLiveZ").textContent, "+10");
    equal(document.getElementById("studioLiveAngle").textContent, "-2.4");
    if (document.getElementById("studioAbsoluteInclination")) {
      throw new Error("Extra target inclination sentence should not be displayed.");
    }
    for (const property of ["fontSize", "fontWeight", "color"]) {
      equal(getComputedStyle(document.getElementById("backrest"))[property],
        getComputedStyle(document.getElementById("seatHeight"))[property]);
      equal(getComputedStyle(document.getElementById("backrest"))[property],
        getComputedStyle(document.getElementById("seatDepth"))[property]);
    }
    equal(document.querySelectorAll(".studio-inclination").length, 0);
    equal(document.getElementById("studio-delta-recline").textContent, "more upright from nominal comfort reference (illustrative, not fitted)");
    api.update(178);
    equal(document.getElementById("backrest").textContent, "—");
    equal(document.getElementById("studioLiveAngle").textContent, "—");
    if (!document.getElementById("studio-delta-recline").textContent.includes("Unavailable")) {
      throw new Error("Missing backrest estimate must be explicit.");
    }
  });
  check("All four figures use gray reference and orange target without dashed lines or arrows", () => {
    api.update(185);
    if (document.querySelector("#seatStudio svg marker, #seatStudio .studio-vector")) {
      throw new Error("Movement arrows remain in seat diagrams.");
    }
    const figures = document.querySelectorAll(".movement-figures svg, #studioHero");
    equal(figures.length, 4);
    for (const svg of figures) {
      if (svg.querySelector("[stroke-dasharray]")) throw new Error("Dashed line remains.");
      const ghost = svg.querySelector(".studio-ghost"), solid = svg.querySelector(".studio-solid");
      if (!getComputedStyle(ghost).filter.includes("grayscale(1)")) {
        throw new Error("Reference is not gray.");
      }
      const cushion = solid.querySelector(".studio-seat-base path:nth-child(3)");
      if (!cushion.getAttribute("fill").includes("-orange-leather")) {
        throw new Error("Target is not filled with orange material.");
      }
    }
  });
  check("Nominal reference schematic is confined to reference details", () => {
    const schema = document.getElementById("studioReferenceSchema");
    if (!schema || !schema.closest('[data-studio-view="reference"]')) {
      throw new Error("Nominal reference schematic is missing from reference details.");
    }
    if (!schema.textContent.includes("Reference X 127.5 mm") ||
        !schema.textContent.includes("Reference Z 40 mm") ||
        !schema.textContent.includes("Reference 25°")) {
      throw new Error("Nominal reference coordinates are missing.");
    }
    if (document.querySelector(".studio-details .studio-disclaimer")) {
      throw new Error("Removed seat disclaimer should not be displayed.");
    }
    equal(document.querySelectorAll(".movement-figures > article").length, 3);
    if (!document.getElementById("studioHero") || !document.getElementById("preview")) {
      throw new Error("Combined seat animation should remain available.");
    }
  });
  check("Steering wheels match and remain fixed in all five views", () => {
    api.update(178, 55);
    const wheels = [...document.querySelectorAll("#seatStudio .studio-steering-wheel")];
    equal(wheels.length, 5);
    const before = wheels.map(wheel => wheel.outerHTML);
    for (const wheel of wheels) {
      equal(wheel.getAttribute("transform"), "translate(675 195)");
      if (!wheel.querySelector(".studio-wheel-rim") || !wheel.querySelector(".studio-wheel-hub")) {
        throw new Error("Wheel rim or hub is missing.");
      }
      const svg = wheel.closest("svg");
      for (const element of wheel.querySelectorAll("[fill], [stroke]")) {
        for (const attribute of ["fill", "stroke"]) {
          const match = element.getAttribute(attribute)?.match(/^url\(#(.+)\)$/);
          if (match && !svg.querySelector(`[id="${match[1]}"]`)) throw new Error("Missing wheel material.");
        }
      }
      const referenceWheel = document.querySelector("#studioReferenceSchema .studio-steering-wheel");
      const heroWheel = document.querySelector("#studioHero .studio-steering-wheel");
      if (referenceWheel.outerHTML.replace(/reference-/g, "hero-") !== heroWheel.outerHTML) {
        throw new Error("Reference steering wheel differs from the animation.");
      }
    }
    api.update(185, 45);
    equal([...document.querySelectorAll("#seatStudio .studio-steering-wheel")].map(wheel => wheel.outerHTML), before);
  });
  check("View buttons show the selected content without resetting the seat", () => {
    api.update(185, 55);
    const target = document.getElementById("seatDepth").textContent;
    for (const screen of ["capture", "adjustments", "animation", "reference"]) {
      document.querySelector(`[data-screen="${screen}"]`).click();
      equal(document.querySelector(`[data-screen="${screen}"]`).getAttribute("aria-pressed"), "true");
      equal(document.getElementById("captureScreen").hidden, screen !== "capture");
      equal(document.getElementById("seatStudio").hidden, screen === "capture");
      for (const panel of document.querySelectorAll("[data-studio-view]")) {
        equal(panel.hidden, panel.dataset.studioView !== screen);
      }
      equal(document.getElementById("seatDepth").textContent, target);
    }
    document.querySelector('[data-screen="capture"]').click();
  });
  return checks;
}
