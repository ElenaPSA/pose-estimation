/* One coordinate model drives the cards and animated seat. */
const SeatStudio = (() => {
  const limits = Object.freeze({x: [0, 255], z: [0, 80], recline: [15, 55]});
  const reference = Object.freeze({x: 127.5, z: 40, recline: 25});
  const fitAnchors = Object.freeze([
    Object.freeze({height: 163, deltaX: 50, deltaZ: -10}),
    Object.freeze({height: 185, deltaX: -50, deltaZ: 10})
  ]);
  const fittingRange = Object.freeze([fitAnchors[0].height, fitAnchors[1].height]);
  const clamp = (n, low, high) => Math.max(low, Math.min(high, n));
  const signed = n => `${n > 0 ? "+" : ""}${Number(n.toFixed(1))}`;
  const nominalTorsoCm = 50;
  let target = null;
  let frame = null;
  let generation = 0;
  let animationComplete = null;
  const reducedMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

  function compute(height, torsoCm = null) {
    if (!Number.isFinite(height) || height < 80 || height > 260) {
      throw new Error("Height must be between 80 and 260 cm.");
    }
    const [shorter, taller] = fitAnchors;
    const fraction = (height - shorter.height) / (taller.height - shorter.height);
    const deltaX = shorter.deltaX + fraction * (taller.deltaX - shorter.deltaX);
    const deltaZ = shorter.deltaZ + fraction * (taller.deltaZ - shorter.deltaZ);
    let recline = null;
    if (Number.isFinite(torsoCm) && torsoCm / height >= .15 && torsoCm / height <= .5) {
      const nominalProjection = nominalTorsoCm * Math.sin(reference.recline * Math.PI / 180);
      const angle = Math.asin(clamp(nominalProjection / torsoCm, 0, 1)) * 180 / Math.PI;
      recline = clamp(angle, reference.recline - 5, reference.recline + 5);
    }
    return {
      x: clamp(reference.x + deltaX, limits.x[0], limits.x[1]),
      z: clamp(reference.z + deltaZ, limits.z[0], limits.z[1]),
      recline
    };
  }
  function supports(height) {
    return Number.isFinite(height) && height >= 80 && height <= 260;
  }
  function delta(value, from = reference) {
    return {x: value.x - from.x, z: value.z - from.z,
      recline: value.recline === null ? null : value.recline - from.recline};
  }
  function transform(value) {
    return {x: 340 + value.x * .75, y: 400 - value.z * .75,
      angle: -(value.recline === null ? reference.recline : value.recline)};
  }
  function definitions(prefix) {
    return `<defs>
      <linearGradient id="${prefix}-leather" x1="0" x2="1" y2=".6">
        <stop stop-color="#30343a"/><stop offset=".38" stop-color="#899099"/>
        <stop offset=".65" stop-color="#626971"/><stop offset="1" stop-color="#292e34"/>
      </linearGradient>
      <linearGradient id="${prefix}-bolster" x2="1" y2="1">
        <stop stop-color="#989ea6"/><stop offset=".45" stop-color="#555c64"/><stop offset="1" stop-color="#282d33"/>
      </linearGradient>
      <linearGradient id="${prefix}-metal"><stop stop-color="#172435"/><stop offset=".5" stop-color="#7c8b9e"/><stop offset="1" stop-color="#243346"/></linearGradient>
      <linearGradient id="${prefix}-wheel-rim" x1="0" y1="0" x2="1" y2="1">
        <stop stop-color="#b4c2d0"/><stop offset=".3" stop-color="#526174"/>
        <stop offset=".65" stop-color="#263447"/><stop offset="1" stop-color="#8597ac"/>
      </linearGradient>
      <linearGradient id="${prefix}-wheel-hub" x1="0" y1="0" x2="1" y2=".7">
        <stop stop-color="#667a91"/><stop offset=".5" stop-color="#24364c"/><stop offset="1" stop-color="#111d2c"/>
      </linearGradient>
      <linearGradient id="${prefix}-orange-leather" x1="0" x2="1" y2=".6">
        <stop stop-color="#713310"/><stop offset=".38" stop-color="#ffb16d"/>
        <stop offset=".65" stop-color="#db762c"/><stop offset="1" stop-color="#71300f"/>
      </linearGradient>
      <linearGradient id="${prefix}-orange-bolster" x2="1" y2="1">
        <stop stop-color="#ffc38c"/><stop offset=".45" stop-color="#d6762e"/><stop offset="1" stop-color="#79330e"/>
      </linearGradient>
      <linearGradient id="${prefix}-orange-metal"><stop stop-color="#713a1e"/><stop offset=".5" stop-color="#e6a16c"/><stop offset="1" stop-color="#84441f"/></linearGradient>
      <radialGradient id="${prefix}-halo"><stop stop-color="#f69138" stop-opacity=".14"/><stop offset="1" stop-color="#f69138" stop-opacity="0"/></radialGradient>
      <pattern id="${prefix}-grid" width="32" height="32" patternUnits="userSpaceOnUse"><path d="M32 0H0V32" fill="none" stroke="#90a8c0" stroke-opacity=".06"/></pattern>
      <filter id="${prefix}-shadow" x="-60%" y="-30%" width="220%" height="190%"><feDropShadow dx="0" dy="14" stdDeviation="9" flood-opacity=".5"/></filter>
    </defs>`;
  }
  function seat(prefix, id, value, ghost = false) {
    const t = transform(value);
    const material = ghost ? prefix : `${prefix}-orange`;
    return `<g id="${id}" class="${ghost ? "studio-ghost" : "studio-solid"}" transform="translate(${t.x} ${t.y})">
      <g class="studio-seat-base">
        <path d="M-21 17H159L151 25H-28Z" fill="url(#${material}-metal)" stroke="#8193aa" stroke-width="1"/>
        <path d="M-12 0L-18 17M133 0L140 17" stroke="#667b94" stroke-width="7"/>
        <path d="M-26 -34Q33 -57 109 -48Q151 -47 165 -29L164 -12Q72 1 -21 -7Z" fill="url(#${material}-leather)" stroke="#7d8ba0"/>
        <path d="M-17 -34Q41 -47 106 -40Q135 -40 151 -28Q76 -17 -16 -22Z" fill="url(#${material}-bolster)" stroke="#8a97aa" stroke-opacity=".6"/>
        <path d="M20 -33Q75 -41 125 -32M22 -29Q72 -36 120 -28" fill="none" stroke="#c0cbda" stroke-opacity=".22"/>
        <path d="M-13 -6L147 -9L143 4H-10Z" fill="#162435" stroke="#51657c"/>
        <rect x="70" y="-3" width="27" height="6" rx="3" fill="#62748a"/><circle cx="115" cy="-1" r="5" fill="#283c54" stroke="#6b819b"/>
      </g>
      <g class="studio-backrest" transform="rotate(${t.angle} 0 -18)">
        <path d="M-25 -23Q-42 -52 -35 -112L-29 -201Q-26 -220 -6 -223Q18 -221 23 -201L26 -104Q30 -49 12 -23Z" fill="url(#${material}-leather)" stroke="#8594a8"/>
        <path d="M-26 -50Q-31 -107 -22 -196Q-20 -209 -7 -209Q8 -207 11 -193L15 -103Q18 -68 7 -47Z" fill="url(#${material}-bolster)" stroke="#8492a5" stroke-opacity=".5"/>
        <path d="M-14 -191L-20 -85Q-21 -69 -14 -55M3 -192L7 -92" fill="none" stroke="#bfccdb" stroke-opacity=".32"/>
        <path d="M-18 -128Q-4 -132 12 -125M-21 -117Q-3 -121 14 -114" stroke="#152336" fill="none"/>
        <path d="M-16 -224V-238M2 -224V-238" stroke="#94a7ba" stroke-width="4"/>
        <path d="M-29 -236L-28 -268Q-25 -278 -7 -279Q13 -278 16 -267L19 -238Q-3 -228 -29 -236Z" fill="url(#${material}-leather)" stroke="#8998aa"/>
        <path d="M-21 -242L-20 -266Q-5 -274 8 -264L10 -244" fill="none" stroke="#c1ccda" stroke-opacity=".3"/>
      </g>
      <circle cx="0" cy="-18" r="17" fill="url(#${material}-metal)" stroke="#92a1b5"/>
      <circle cx="0" cy="-18" r="10" fill="#1b2b40" stroke="#596f89"/>
    </g>`;
  }
  function steeringWheel(prefix) {
    return `<g class="studio-steering-wheel" transform="translate(675 195)" role="img" aria-label="Illustrative steering wheel and column, fixed ahead of the seat">
      <path d="M4 5L32 64" fill="none" stroke="#0a111c" stroke-width="18" stroke-linecap="round"/>
      <path d="M4 5L32 64" fill="none" stroke="url(#${prefix}-metal)" stroke-width="12" stroke-linecap="round"/>
      <path d="M9 11L34 63" fill="none" stroke="#8b9bb0" stroke-width="2" stroke-linecap="round" opacity=".6"/>
      <g transform="rotate(18)">
        <ellipse cx="4" cy="1" rx="22" ry="54" fill="none" stroke="#070e18" stroke-width="12"/>
        <ellipse class="studio-wheel-rim" rx="22" ry="54" fill="none" stroke="url(#${prefix}-wheel-rim)" stroke-width="9"/>
        <path d="M-17 -32Q-27 -6 -17 25M14 -41Q23 -18 21 8" fill="none" stroke="#c9d5e0" stroke-width="1.5" stroke-linecap="round" opacity=".55"/>
        <path d="M-19 -13L-5 -3L5 -3L19 -13L17 -3L6 6L6 42L-6 42L-6 6L-17 -3Z"
          fill="url(#${prefix}-metal)" stroke="#798ba1" stroke-width="1"/>
        <path d="M-4 12V35M4 12V35" stroke="#b5c1ce" stroke-width="1" opacity=".55"/>
        <ellipse class="studio-wheel-hub" rx="11" ry="15" fill="url(#${prefix}-wheel-hub)" stroke="#93a5b9" stroke-width="1"/>
        <path d="M-6 -7Q0 -11 6 -7M-5 8H5" fill="none" stroke="#a7b8ca" stroke-width="1" opacity=".6"/>
        <path d="M-19 -3L-13 1M13 1L19 -3" stroke="#d2dce6" stroke-width="2" stroke-linecap="round"/>
      </g>
    </g>`;
  }
  function scene(prefix, value, ghostValue = reference) {
    return `${definitions(prefix)}
      <rect width="900" height="520" fill="#0b1523"/>
      <rect width="900" height="520" fill="url(#${prefix}-grid)"/>
      <ellipse cx="470" cy="275" rx="310" ry="220" fill="url(#${prefix}-halo)"/>
      <ellipse cx="470" cy="435" rx="250" ry="15" fill="#020710" opacity=".6"/>
      <path d="M305 433H735M305 440H735" stroke="#41556e" stroke-width="3"/>
      <path d="M340 459V469M531.25 459V469M340 464H531.25" fill="none" stroke="#6b829d" stroke-width="1"/>
      <path d="M340 400H735M340 415V65" stroke="#5d8fe8" stroke-opacity=".65" fill="none"/>
      <path d="M727 395L735 400L727 405M335 73L340 65L345 73" fill="none" stroke="#7aa6ef"/>
      <path d="M295 453H755" stroke="#8b9bb1" opacity=".3"/>
      <g fill="#90a4bf" font-family="Arial" font-size="12">
        <text x="340" y="487" text-anchor="middle">0 / REAR STOP</text><text x="531.25" y="487" text-anchor="middle">255 mm / FRONT STOP</text>
        <text x="688" y="388">+X</text><text x="352" y="78">+Z</text>
      </g>
      ${steeringWheel(prefix)}
      ${seat(prefix, `${prefix}-ghost`, ghostValue, true)}
      ${seat(prefix, `${prefix}-seat`, value)}
      <circle cx="340" cy="400" r="4" fill="#6cc7c2"/>
      <text x="330" y="420" text-anchor="end" fill="#6cc7c2" font-family="Arial" font-size="11">O / REAR + LOW</text>
      <text x="145" y="456" fill="#8196b1" font-family="Arial" font-size="10">Floor: Z = -70 mm</text>`;
  }

  function referenceSchema() {
    const t = transform(reference), hingeY = t.y - 18;
    const tip = angle => {
      const radians = angle * Math.PI / 180;
      return {x: t.x - 230 * Math.sin(radians), y: hingeY - 230 * Math.cos(radians)};
    };
    const upright = tip(limits.recline[0]), reclined = tip(limits.recline[1]), nominal = tip(reference.recline);
    const rear = transform({...reference, x: limits.x[0]});
    const front = transform({...reference, x: limits.x[1]});
    const low = transform({...reference, z: limits.z[0]});
    const high = transform({...reference, z: limits.z[1]});
    return `<svg id="studioReferenceSchema" viewBox="130 40 730 490" role="img" aria-labelledby="studioSchemaTitle studioSchemaDescription">
      <title id="studioSchemaTitle">Seat travel limits and nominal comfort reference</title>
      <desc id="studioSchemaDescription">Gray seat at X ${reference.x} mm, Z ${reference.z} mm, ${70 + reference.z} mm above floor, recline ${reference.recline} degrees from vertical. X range ${limits.x.join(" to ")} mm, Z range ${limits.z.join(" to ")} mm, recline range ${limits.recline.join(" to ")} degrees. Seat faces right.</desc>
      ${definitions("reference")}
      <rect x="130" y="40" width="730" height="490" rx="12" fill="#0b1523"/>
      <rect x="130" y="40" width="730" height="490" fill="url(#reference-grid)"/>
      <path d="M305 433H735M305 440H735" stroke="#41556e" stroke-width="3"/>
      <path d="M295 453H755" stroke="#8b9bb1" opacity=".4"/>
      ${steeringWheel("reference")}
      <path d="M${t.x} ${hingeY}V95 M${t.x} ${hingeY}L${upright.x} ${upright.y} M${t.x} ${hingeY}L${reclined.x} ${reclined.y}
        M${upright.x} ${upright.y}A230 230 0 0 0 ${reclined.x} ${reclined.y}" fill="none" stroke="#71859d" stroke-width="1.5"/>
      ${seat("reference", "reference-nominal", reference, true)}
      <g fill="none" stroke="#aab4bf" stroke-width="1.5">
        <path d="M${rear.x} 447V477M${front.x} 447V477M${rear.x} 469H${front.x}"/>
        <path d="M628 ${high.y}H650M628 ${low.y}H650M642 ${high.y}V${low.y}"/>
        <path d="M${t.x} ${t.y}H620M${t.x} ${t.y}V469"/>
      </g>
      <g fill="#ffb16d" stroke="#ffb16d">
        <circle cx="${t.x}" cy="${t.y}" r="5"/><circle cx="${t.x}" cy="469" r="5"/>
        <circle cx="642" cy="${t.y}" r="5"/><circle cx="${nominal.x}" cy="${nominal.y}" r="5"/>
      </g>
      <g font-family="Segoe UI, sans-serif" font-size="15" fill="#c6d2e0">
        <text x="690" y="82">FRONT / WHEEL</text>
        <text x="${t.x + 12}" y="102">Vertical / 0&#176;</text>
        <text x="${upright.x - 8}" y="${upright.y - 12}" text-anchor="end">${limits.recline[0]}&#176; min</text>
        <text x="${reclined.x - 12}" y="${reclined.y + 5}" text-anchor="end">${limits.recline[1]}&#176; max</text>
        <text x="${rear.x}" y="500" text-anchor="middle">${limits.x[0]} mm / rear</text>
        <text x="${front.x}" y="500" text-anchor="middle">${limits.x[1]} mm / front</text>
        <text x="662" y="${high.y + 5}">Z ${limits.z[1]} / 150 mm floor</text>
        <text x="662" y="${low.y + 5}">Z ${limits.z[0]} / 70 mm floor</text>
        <text x="670" y="449">Vehicle floor</text>
      </g>
      <g font-family="Segoe UI, sans-serif" font-size="16" font-weight="600" fill="#ffb16d">
        <text x="160" y="100">RECLINE RANGE</text>
        <text x="160" y="124">${limits.recline[0]}&ndash;${limits.recline[1]}&#176; from vertical</text>
        <text x="${nominal.x - 15}" y="${nominal.y + 5}" text-anchor="end">Reference ${reference.recline}&#176;</text>
        <text x="662" y="306">HEIGHT / mm</text>
        <text x="662" y="${t.y + 5}">Reference Z ${reference.z} mm</text>
        <text x="160" y="420">DEPTH / ${limits.x[0]}&ndash;${limits.x[1]} mm</text>
        <text x="${t.x}" y="525" text-anchor="middle">Reference X ${reference.x} mm</text>
      </g>
    </svg>`;
  }

  function mount() {
    document.getElementById("seatStudio").innerHTML = `
      <div class="studio-heading"><div><div class="studio-eyebrow">CONFIGURATION VECTOR / KINEMATIC STUDIO</div>
      <h2>One seat. One reference.</h2><p>Every view faces the wheel on the right. Every number uses the same measured travel.</p></div>
      <span class="studio-badge">SIMULATION ONLY</span></div>
      <div id="seatFitModelStatus" class="studio-disclaimer" role="status"></div>
      <div class="movement-figures" data-studio-view="adjustments">
        ${[["recline", "01", "Backrest Inclination", "backrest", "&#176;", "15&ndash;55&deg; from vertical"],
          ["z", "02", "Seat Height", "seatHeight", "mm", "0&ndash;80 mm above lowest"],
          ["x", "03", "Seat Depth", "seatDepth", "mm", "0&ndash;255 mm from rear stop"]].map(([key, n, name, id, unit, range]) => `
          <article class="movement-row"><div class="studio-card-heading"><span>${n} / ${key === "recline" ? "RECLINE" : key.toUpperCase()+" AXIS"}</span><span class="studio-direction">FRONT &rarr;</span></div>
          <svg id="studio-card-${key}" viewBox="130 40 630 480" role="img" aria-label="${name}: orange target overlaid on gray nominal comfort reference, whole seat facing right"></svg>
          <div class="studio-card-copy"><span class="movement-label">${name}</span><output aria-label="${name} movement from nominal comfort reference"><b id="${id}"></b><small${key === "recline" ? ' class="angle-unit"' : ""}>${unit}</small></output>
          <span id="studio-delta-${key}" class="studio-delta"></span><span id="studio-target-${key}" class="studio-range"></span>
          <span class="studio-range">Vehicle travel limit: ${range}</span>
          <span class="studio-card-legend"><i></i> Gray: nominal comfort reference <i class="studio-target-key"></i> Orange: simulated seat</span></div></article>`).join("")}
      </div>
      <div class="studio-hero" data-studio-view="animation" hidden>
        <div class="studio-hero-top"><div><div class="studio-eyebrow">COMBINED MOTION</div><h3>Prototype target, in motion.</h3></div><button id="preview" type="button" class="studio-play">PLAY ADJUSTMENT <span>&rarr;</span></button></div>
        <div class="studio-timeline" aria-label="Animation stages"><span data-stage="0">01 &nbsp; Fore / aft</span><span data-stage="1">02 &nbsp; Elevation</span><span data-stage="2">03 &nbsp; Recline</span><span data-stage="3">04 &nbsp; Final pose</span></div>
        <div class="studio-stage"><span class="studio-stage-label">REAR &nbsp; &larr;</span><span class="studio-stage-label studio-front">FRONT / WHEEL &nbsp; &rarr;</span>
          <svg id="studioHero" viewBox="130 40 730 470" role="img" aria-label="Whole-seat animation, positive X right toward the wheel, positive Z up"></svg>
          <div class="studio-live" aria-label="Animated movement from nominal comfort reference"><span>&Delta;X <b id="studioLiveX"></b> mm</span><span>&Delta;Z <b id="studioLiveZ"></b> mm</span><span>&Delta;&theta; <b id="studioLiveAngle"></b>&deg;</span></div>
        </div>
        <div class="studio-progress"><i id="studioProgress"></i></div>
        <div class="studio-legend"><span><i></i> Gray: nominal comfort reference</span><span><i></i> Orange: simulated seat</span><span>SCHEMATIC &middot; NOT TO SCALE</span></div>
        <div id="seatAnimationStatus" role="status" aria-live="polite"></div>
      </div>
      <div class="studio-details" data-studio-view="reference" hidden><div class="studio-reference-schema">${referenceSchema()}</div>
        <div><div class="studio-eyebrow">NOMINAL COMFORT REFERENCE / ASSUMED</div><h3>Coordinates without ambiguity.</h3>
        <p>Nominal seat: <b>X ${reference.x} mm</b>, <b>Z ${reference.z} mm</b> (${70 + reference.z} mm above the floor), and <b>${reference.recline}&deg;</b> from vertical. This is an assumed demonstration position, not a measured or validated comfort setting.</p>
        <p>Fixed origin <b>O</b>: your marked cushion-bottom point at the <b>fully rearward, lowest</b> position.
        X increases toward the wheel on the right. Z increases upward. Y points vehicle-left; there is no lateral motion.</p>
        <p>Floor-to-marker height = <b>70 + Z mm</b> (70&ndash;150 mm). Recline is measured from vertical, increasing rearward.
        The marked point is not a measured automotive SRP. Rail, hinge and upholstery geometry are illustrative.</p></div>
        <div><div class="studio-eyebrow">TARGETS &ne; MOVEMENT COMMANDS</div><p id="seatTargetDescription"></p>
        <p>Signed movements and live animation values use the nominal comfort reference, not the rear/lowest coordinate origin.
        Each card shows only that axis changing; other axes stay at the reference. The hero combines all three, starting at zero movement.
        The gray reference stays behind the orange simulated seat; they coincide when there is no change.</p></div></div>`;
    document.getElementById("preview").addEventListener("click", play);
  }
  function position(prefix, value) {
    const t = transform(value);
    document.getElementById(`${prefix}-seat`).setAttribute("transform", `translate(${t.x} ${t.y})`);
    document.querySelector(`#${prefix}-seat .studio-backrest`).setAttribute("transform", `rotate(${t.angle} 0 -18)`);
  }
  function live(value) {
    const change = delta(value);
    document.getElementById("studioLiveX").textContent = signed(change.x);
    document.getElementById("studioLiveZ").textContent = signed(change.z);
    document.getElementById("studioLiveAngle").textContent =
      target.recline === null ? "—" : signed(change.recline);
  }
  function stop() {
    if (animationComplete) animationComplete(false);
    animationComplete = null;
    generation++;
    if (frame !== null) cancelAnimationFrame(frame);
    frame = null;
    document.getElementById("preview").textContent = "PLAY ADJUSTMENT \u2192";
  }
  function stages(stage) {
    document.querySelectorAll("[data-stage]").forEach((el, i) => {
      el.classList.toggle("active", i === stage);
      el.classList.toggle("complete", i < stage);
    });
  }
  function update(height, torsoCm = null) {
    target = compute(height, torsoCm);
    stop();
    const available = target !== null;
    document.getElementById("preview").disabled = !available;
    if (!available) {
      document.getElementById("seatAnimationStatus").textContent =
        "No seat movement estimate: height is outside the supported input range.";
      return;
    }
    const withinFitData = height >= fittingRange[0] && height <= fittingRange[1];
    const fitStatus = document.getElementById("seatFitModelStatus");
    fitStatus.hidden = withinFitData;
    fitStatus.textContent = withinFitData ? "" :
      `Seat X/Z extrapolated beyond the two reported seat trials (${fittingRange[0]} and ${fittingRange[1]} cm), clipped to travel limits; unvalidated.`;
    const change = delta(target);
    for (const [axis, id] of [["x","seatDepth"],["z","seatHeight"],["recline","backrest"]]) {
      const unavailable = change[axis] === null;
      document.getElementById(id).textContent = unavailable ? "—" : signed(change[axis]);
      document.getElementById(`studio-card-${axis}`).innerHTML = scene(`card-${axis}`,
        {...reference, [axis]: unavailable ? reference[axis] : target[axis]}, reference);
      const direction = axis === "x" ? (change.x < 0 ? "rearward" : "forward") :
        axis === "z" ? (change.z < 0 ? "downward" : "upward") :
        (change.recline < 0 ? "more upright" : "more reclined");
      const description = change[axis] === 0
        ? "No movement from nominal comfort reference" : `${direction} from nominal comfort reference`;
      document.getElementById(`studio-delta-${axis}`).textContent = unavailable
        ? "Unavailable: a usable detected torso estimate is required"
        : axis === "recline" ? `${description} (illustrative, not fitted)` : description;
      const unit = axis === "recline" ? "\u00b0" : " mm";
      document.getElementById(`studio-target-${axis}`).textContent = unavailable
        ? "Backrest shown at reference for illustration only; no estimate."
        :
        `Reference ${reference[axis]}${unit} \u2192 target ${Number(target[axis].toFixed(1))}${unit} (absolute)`;
    }
    document.getElementById("studioHero").innerHTML = scene("hero", target);
    document.getElementById("seatTargetDescription").textContent =
      `Prototype target: X ${Number(target.x.toFixed(1))} mm, Z ${Number(target.z.toFixed(1))} mm; ${
        target.recline === null ? "backrest estimate unavailable" :
          `illustrative backrest ${Number(target.recline.toFixed(1))}\u00b0 from vertical`}.`;
    document.getElementById("seatAnimationStatus").textContent = withinFitData
      ? "Interpolated target ready within reported height anchors. Starting seat position is assumed."
      : "Extrapolated target clipped to travel limits; estimate is unvalidated beyond reported heights.";
    document.getElementById("studioProgress").style.width = "0%";
    stages(-1);
    live(target);
  }
  function play() {
    if (!target) return Promise.resolve(false);
    stop();
    const completion = new Promise(resolve => { animationComplete = resolve; });
    const run = generation;
    const end = {...target, recline: target.recline === null ? reference.recline : target.recline};
    const change = delta(end);
    const finish = () => {
      if (run !== generation) return;
      frame = null;
      position("hero", end);
      live(end);
      stages(3);
      document.getElementById("studioProgress").style.width = "100%";
      document.getElementById("preview").textContent = "REPLAY ADJUSTMENT \u21bb";
      document.getElementById("seatAnimationStatus").textContent =
        `Prototype model complete. Movements from assumed reference: X ${signed(change.x)} mm, Z ${signed(change.z)} mm. ${
          target.recline === null ? "Backrest estimate unavailable; displayed at reference." :
            `Illustrative backrest movement ${signed(change.recline)}°.`} No real seat movement.`;
      animationComplete(true);
      animationComplete = null;
    };
    if (reducedMotion()) { finish(); return completion; }
    const duration = 4200;
    let started;
    position("hero", reference);
    live(reference);
    stages(0);
    document.getElementById("preview").textContent = "RESTART ANIMATION \u21bb";
    document.getElementById("seatAnimationStatus").textContent = "Playing demo: fore/aft, elevation, then recline. No vehicle commands.";
    const tick = now => {
      if (run !== generation) return;
      if (started === undefined) started = now;
      const progress = Math.min(1, (now - started) / duration);
      const axisProgress = offset => {
        const t = clamp(progress * 3 - offset, 0, 1);
        return t * t * (3 - 2 * t);
      };
      const value = {
        x: reference.x + change.x * axisProgress(0),
        z: reference.z + change.z * axisProgress(1),
        recline: reference.recline + change.recline * axisProgress(2)
      };
      position("hero", value);
      live(value);
      stages(Math.min(2, Math.floor(progress * 3)));
      document.getElementById("studioProgress").style.width = `${progress * 100}%`;
      if (progress === 1) finish();
      else frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return completion;
  }
  return {mount, update, play, stop, compute, delta, transform, limits, reference, fitAnchors, fittingRange, supports};
})();
