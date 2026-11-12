function runSeatStudioChecks(api) {
  const checks = [];
  function check(name, callback) { callback(); checks.push(name); }
  function equal(value, expected) {
    if (JSON.stringify(value) !== JSON.stringify(expected)) {
      throw new Error(`Expected ${JSON.stringify(expected)}, got ${JSON.stringify(value)}`);
    }
  }
  check("Rebased small, medium and large targets", () => {
    equal(api.compute(160), {x: 255, z: 1, recline: 25});
    equal(api.compute(178), {x: 240, z: 20, recline: 25});
    equal(api.compute(190), {x: 227, z: 35, recline: 27});
  });
  check("Every height in the supported range stays inside measured limits", () => {
    for (let h = 120; h <= 230; h += .25) {
      const value = api.compute(h);
      for (const axis of ["x", "z", "recline"]) {
        if (value[axis] < api.limits[axis][0] || value[axis] > api.limits[axis][1]) {
          throw new Error(`Out of bounds: ${h}, ${axis}`);
        }
      }
    }
  });
  check("Invalid heights are errors, never default targets", () => {
    for (const value of [119, 231, NaN, Infinity, "178", null, undefined]) {
      let failed = false;
      try { api.compute(value); } catch { failed = true; }
      if (!failed) throw new Error(`Accepted invalid input ${value}`);
    }
  });
  check("Positive X moves right, positive Z up, recline rearward left", () => {
    const low = api.transform({x: 0, z: 0, recline: 15});
    const high = api.transform({x: 255, z: 80, recline: 55});
    if (high.x <= low.x || high.y >= low.y || high.angle >= low.angle) {
      throw new Error("Screen orientation disagrees with vehicle coordinates.");
    }
    equal(high.x - low.x, 191.25);
    equal(low.y - high.y, 60);
  });
  check("Movements are target minus the nominal comfort reference", () => {
    equal(api.reference, {x: 127.5, z: 40, recline: 25});
    equal(api.delta(api.compute(160)), {x: 127.5, z: -39, recline: 0});
    equal(api.delta(api.compute(178)), {x: 112.5, z: -20, recline: 0});
    equal(api.delta(api.compute(190)), {x: 99.5, z: -5, recline: 2});
    equal(api.delta(api.reference), {x: 0, z: 0, recline: 0});
    equal(api.delta({x: 0, z: 0, recline: 15}, {x:255, z:80, recline:55}),
      {x: -255, z: -80, recline: -40});
  });
  check("Cards show signed movements and fixed overlaid reference seats", () => {
    api.update(190);
    for (const [axis, id, expected] of [["x", "seatDepth", "+99.5"], ["z", "seatHeight", "-5"], ["recline", "backrest", "+2"]]) {
      equal(document.getElementById(id).textContent, expected);
      const card = document.getElementById(`studio-card-${axis}`);
      const ghost = card.querySelector(".studio-ghost"), solid = card.querySelector(".studio-solid");
      const t = api.transform(api.reference);
      equal(ghost.getAttribute("transform"), `translate(${t.x} ${t.y})`);
      equal(ghost.querySelector(".studio-backrest").getAttribute("transform"), `rotate(${t.angle} 0 -18)`);
      if (!(ghost.compareDocumentPosition(solid) & Node.DOCUMENT_POSITION_FOLLOWING)) {
        throw new Error("Orange simulated seat must be drawn over the gray reference.");
      }
      if (!document.getElementById(`studio-target-${axis}`).textContent.includes("Reference")) {
        throw new Error("Missing absolute reference-to-target caption.");
      }
    }
    equal(document.getElementById("studioLiveX").textContent, "+99.5");
    equal(document.getElementById("studioLiveZ").textContent, "-5");
    equal(document.getElementById("studioLiveAngle").textContent, "+2");
    api.update(178);
    equal(document.getElementById("backrest").textContent, "0");
    equal(document.getElementById("studio-delta-recline").textContent, "No movement from nominal comfort reference");
  });
  check("All four figures use gray reference and orange target without dashed lines or arrows", () => {
    api.update(190);
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
  check("Reference schematic precedes the three unchanged movement cards", () => {
    const schema = document.getElementById("studioReferenceSchema");
    if (!schema || !schema.querySelector("#reference-nominal")) throw new Error("Missing reference schematic.");
    const description = schema.querySelector("desc").textContent;
    for (const value of ["127.5 mm", "40 mm", "110 mm", "25 degrees", "0 to 255", "0 to 80", "15 to 55"]) {
      if (!description.includes(value)) throw new Error(`Missing reference dimension ${value}`);
    }
    equal(document.querySelectorAll(".movement-figures > article").length, 3);
    if (!(schema.compareDocumentPosition(document.querySelector(".movement-figures")) & Node.DOCUMENT_POSITION_FOLLOWING)) {
      throw new Error("Reference schematic must precede the movement figures.");
    }
  });
  return checks;
}
