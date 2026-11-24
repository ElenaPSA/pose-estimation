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
