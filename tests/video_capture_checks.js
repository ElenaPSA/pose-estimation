function runVideoCaptureChecks(api) {
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
