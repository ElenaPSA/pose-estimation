from __future__ import annotations

import argparse
import base64
import binascii
import json
import logging
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import ClassVar, cast

import cv2
import numpy as np

from .config import Settings
from .face_detector import FaceHeightDetector
from .models import FramePoseResult, PoseDetection
from .morphology_analyzer import MorphologyAnalyzer
from .pose_estimator import DriverPoseEstimator

logger = logging.getLogger(__name__)

MAX_REQUEST_BYTES = 24 * 1024 * 1024
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_DIMENSION = 6000
ALLOWED_IMAGE_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Driver seat settings prototype</title>
<style>
:root {
  color-scheme: dark;
  --bg: #060b12; --panel: #0d1724; --panel2: #111d2d; --line: #26364a;
  --text: #eaf1fb; --muted: #9aa9bc; --blue: #5d8fe8;
  --cyan: #6cc7c2; --green: #79c58f; --amber: #d8a15a;
  --coral: #f08a6e; --ink: #dfeaf8;
}
* { box-sizing: border-box; }
body {
  margin: 0; min-height: 100vh; color: var(--text); background: var(--bg);
  font: 14px/1.5 "Segoe UI", Arial, sans-serif;
}
button, input { font: inherit; }
button { color: inherit; }
.shell { max-width: 1480px; margin: auto; padding: 26px 34px 32px; }
header { display: flex; justify-content: space-between; align-items: center; gap: 20px;
  padding-bottom: 19px; border-bottom: 1px solid var(--line); }
.brand { display: flex; gap: 13px; align-items: center; }
.brand-mark { width: 43px; height: 43px; border-radius: 10px; background: var(--ink);
  display: grid; place-items: center; color: #0d1724; font-size: 17px; font-weight: 750; }
h1 { margin: 0; font-size: 17px; letter-spacing: .12em; font-weight: 750; }
.sub { margin-top: 3px; color: var(--muted); font-size: 12px; }
.top-right { color: var(--muted); text-align: right; font-size: 11px; }
.prototype { color: #f2d9bc; border: 1px solid #6a4a2f; border-radius: 5px;
  padding: 7px 10px; white-space: nowrap; background: rgba(118, 91, 66, 0.38); font-weight: 750; letter-spacing: .04em; }
.layout { display: grid; grid-template-columns: minmax(340px, .86fr) minmax(450px, 1.4fr);
  gap: 22px; margin-top: 22px; }
.left, .right { display: grid; gap: 15px; align-content: start; }
.card { border: 1px solid var(--line); background: var(--panel);
  border-radius: 9px; padding: 18px 20px; box-shadow: 0 3px 12px rgba(2, 6, 12, 0.5); }
.step { display: flex; gap: 9px; align-items: center; margin: 0 0 14px;
  font-size: 12px; letter-spacing: .08em; font-weight: 750; color: var(--muted); }
.info { margin-left: auto; color: var(--muted); border: 1px solid #475d78;
  border-radius: 4px; width: 18px; height: 18px; text-align: center; }
.info:focus-visible, button:focus-visible, input:focus-visible {
  outline: 3px solid #d47734; outline-offset: 3px;
}
.height-line { display: flex; align-items: baseline; gap: 8px; }
.height-line output { color: var(--ink); font-size: 30px; min-width: 56px; font-weight: 650; }
.height-line span { color: var(--muted); }
input[type=range] { width: 100%; accent-color: var(--blue); cursor: pointer; }
.range-wrap { padding: 0 6px; }
.range-labels { display: flex; justify-content: space-between; padding: 0 4px;
  color: var(--muted); font-size: 10px; }
.profiles, .body-types { display: grid; grid-template-columns: repeat(3, 1fr); gap: 9px; }
.choice { min-height: 88px; border: 1px solid var(--line); border-radius: 7px;
  background: #121f2f; cursor: pointer; padding: 10px 7px; text-align: center;
  transition: .15s ease; }
.choice:hover { border-color: #58779a; background: #18273d; transform: translateY(-1px); }
.choice.active { border: 2px solid var(--blue); background: #182d46; box-shadow: inset 0 -3px 0 var(--blue); }
.person-icon { width: 31px; height: 31px; display: grid; place-items: center;
  margin: 0 auto 4px; border-radius: 50%; color: #eaf1fb; background: var(--blue);
  font-size: 12px; line-height: 1; font-weight: 750; }
.choice strong { display: block; font-size: 12px; font-weight: 700; }
.choice small { display: block; color: var(--muted); font-size: 11px; margin-top: 3px; }
.body-types .choice { min-height: 62px; padding-top: 9px; }
.body-types .person-icon { width: 22px; height: 22px; margin-bottom: 3px;
  font-size: 14px; color: #eaf1fb; background: var(--blue); }
.slider-row { margin: 12px 0 3px; }
.slider-head { display: flex; justify-content: space-between; color: var(--muted); font-size: 12px; }
.slider-head output { color: var(--blue); font-weight: 700; }
.slider-wrap { display: grid; grid-template-columns: 48px 1fr 48px; gap: 7px; align-items: center; }
.slider-wrap small { color: var(--muted); font-size: 11px; }
.captured { margin-top: 12px; padding: 11px; border: 1px dashed #465f7e;
  border-radius: 6px; color: var(--text); font-size: 12px; background: var(--panel2); }
.captured input { display: block; width: 100%; margin-top: 7px; color: var(--text); font-size: 12px; }
.upload-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
.upload-view { min-width: 0; padding: 9px; border: 1px solid var(--line); border-radius: 6px; }
.upload-view label, .calibration label { display: block; font-size: 11px; font-weight: 700; }
.upload-view canvas { display: block; width: 100%; max-height: 240px; margin-top: 8px; object-fit: contain; background: #0b1521; }
.upload-view small, .calibration small { color: var(--muted); font-size: 10px; }
.detection-panel { min-width: 0; padding: 9px; border: 1px solid var(--line); border-radius: 6px; background: #09131f; }
.detection-title { color: var(--ink); font-size: 11px; font-weight: 700; }
.detection-viewport { display: flex; justify-content: center; width: 100%; margin-top: 8px; }
.detection-stage { position: relative; flex: none; max-width: 100%; max-height: 240px; overflow: hidden; background: #0b1521; }
.detection-image { display: block; width: 100%; height: 100%; object-fit: contain; }
.detection-viewport.scanning #detectionImage { animation: grayscale-reveal 5s ease-out both; }
.detection-viewport.detected #detectionImage { filter: grayscale(1); }
.detection-color-image { position: absolute; inset: 0; opacity: 0; clip-path: inset(0 100% 0 0); }
.detection-viewport.detected .detection-color-image { opacity: 1; }
.detection-overlay, .detection-sweep { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; }
.detection-overlay { opacity: 0; }
.detection-viewport.detected .detection-overlay { animation: camera-highlight .35s ease-out both; }
.detection-overlay line, .detection-overlay rect, .detection-overlay circle { fill: none; stroke: #ff861c; stroke-width: .8; vector-effect: non-scaling-stroke; }
.detection-overlay .person-box { stroke-width: 1.4; }
.detection-overlay .camera-body { fill: #ff861c; stroke-width: .5; }
.detection-sweep { top: 0; bottom: auto; height: 2px; opacity: 0; background: #ff2638; box-shadow: 0 0 8px 3px rgba(255, 38, 56, .8); }
.detection-viewport.scanning .detection-sweep { animation: person-scan 5s linear both; }
@keyframes grayscale-reveal { from { filter: grayscale(0); } to { filter: grayscale(1); } }
@keyframes camera-highlight { from { opacity: 0; } to { opacity: 1; } }
@keyframes person-scan {
  0% { transform: translateY(0); opacity: 0; }
  8% { opacity: 1; }
  92% { opacity: 1; }
  100% { transform: translateY(var(--scan-distance)); opacity: .75; }
}
.detection-status { margin-top: 6px; color: var(--muted); font-size: 10px; }
.calibration { display: flex; align-items: end; gap: 10px; margin-top: 12px; }
.calibration input { width: 105px; padding: 7px; border: 1px solid var(--line); border-radius: 4px; }
.upload-submit { margin-top: 12px; padding: 9px 12px; border: 0; border-radius: 5px; background: var(--blue); color: #eaf1fb; font-weight: 700; cursor: pointer; }
.upload-submit:disabled { opacity: .5; cursor: wait; }
.upload-status { margin-top: 9px; font-size: 11px; color: var(--muted); }
.analysis-card { grid-column: 1 / -1; }
.result-title { color: var(--muted); font-size: 11px; letter-spacing: .1em; margin-bottom: 12px; font-weight: 750; }
.parameters { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.parameter { display: grid; grid-template-columns: 25px 1fr; gap: 7px; align-items: center;
  min-height: 58px; padding: 7px; border: 1px solid var(--line); border-radius: 6px; background: var(--panel2); }
.movement-figures { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }
.movement-row { position: relative; min-width: 0; overflow: hidden; aspect-ratio: 1.05 / 1;
  border: 1px solid var(--line); border-radius: 6px; background: #07111c; }
.movement-row svg { position: absolute; inset: 0; display: block; width: 100%; height: 100%; }
.movement-row .movement-art { width: 100%; height: 100%; object-fit: cover; }
.movement-row .movement-arrow { fill: none; stroke: #f27b16; stroke-width: 3; stroke-linecap: round; stroke-linejoin: round; filter: drop-shadow(0 0 2px #15110d); }
.movement-row #horizontalMovementArrow { stroke-width: 2.5; }
.movement-label { position: absolute; top: 7px; left: 7px; z-index: 1; padding: 2px 4px; border-radius: 3px;
  background: rgba(5, 10, 17, .68); color: #ff861c; font-size: 10px; font-weight: 700; }
.movement-copy { position: absolute; right: 9px; bottom: 5px; z-index: 1; display: flex; align-items: baseline; gap: 2px;
  padding: 1px 4px; border-radius: 3px; background: rgba(5, 10, 17, .68); color: #ff861c; font-size: 15px; font-weight: 800; }
.movement-copy .movement-number { color: #ff861c; font-size: 13px; }
.movement-copy .movement-unit { color: #ff861c; font-size: 15px; font-weight: 800; }
.param-icon { color: var(--coral); font-size: 16px; text-align: center; }
.param-name { color: var(--muted); font-size: 11px; }
.param-value { grid-column: 2; text-align: left; font-size: 13px; font-weight: 750; color: var(--ink); }
.param-value small { display: block; font-size: 10px; color: var(--muted); font-weight: 400; }
.cta { display: block; width: min(100%, 430px); margin: 17px auto 0; border: 0;
  border-radius: 6px; padding: 13px 15px; background: #1d2d3c; color: #edf3ff;
  text-align: center; font-weight: 700; cursor: pointer; letter-spacing: .06em; }
.cta:hover { background: #23354d; }
.cta:disabled { opacity: .48; cursor: not-allowed; }
.under-cta { text-align: center; color: var(--muted); font-size: 11px; margin-top: 8px; }
.seat-scheme { position: relative; margin-top: 14px; overflow: hidden;
  border: 1px solid var(--line); border-radius: 8px; background: #0b1521;
  width: min(100%, 620px); margin-left: auto; margin-right: auto; aspect-ratio: 3 / 2; }
.seat-scheme img { position: absolute; inset: 0; display: block; width: 100%;
  height: 100%; object-fit: contain; }
.seat-scheme .seat-proposal { opacity: 0; pointer-events: none; }
#seatProposal { clip-path: polygon(0 55%, 12% 55%, 15% 53%, 23% 53%, 34% 55%, 39% 58%, 43% 60%, 46% 64%, 50% 67%, 55% 70%, 55% 100%, 0 100%); }
#seatBackrestProposal { clip-path: polygon(41% 62%, 35% 58%, 35% 56%, 38% 47%, 42% 38%, 46% 28%, 48% 20%, 50% 14%, 53% 13%, 53% 26%, 51% 35%, 49% 45%, 47% 56%, 46% 62%); transform-origin: 41% 62%; }
.seat-scheme-legend { display: flex; flex-wrap: wrap; gap: 14px;
  color: var(--muted); font-size: 11px; margin-top: 8px; }
.seat-scheme-legend span { display: inline-flex; gap: 6px; align-items: center; }
.seat-scheme-legend i { width: 14px; height: 10px; border-radius: 2px; }
.seat-scheme-status { color: var(--muted); font-size: 11px; margin-top: 6px; }
.foot { margin-top: 20px; display: flex; justify-content: space-between; gap: 15px;
  color: var(--muted); font-size: 11px; border-top: 1px solid var(--line); padding-top: 13px; }
.foot strong { color: var(--coral); font-weight: 750; }
@media (max-width: 900px) {
  .layout { grid-template-columns: 1fr; }
}
@media (max-width: 540px) {
  .shell { padding: 15px 12px; }
  .upload-grid { grid-template-columns: 1fr; }
  header { align-items: flex-start; }
  .top-right { display: none; }
  .movement-figures { grid-template-columns: 1fr; }
  .profiles, .body-types { gap: 5px; }
  .choice { padding-left: 3px; padding-right: 3px; }
  .foot { display: block; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { scroll-behavior: auto !important; }
  .detection-viewport.scanning #detectionImage { animation: none; filter: grayscale(1); }
  .detection-viewport.detected .detection-overlay { animation: none; opacity: 1; }
  .detection-viewport.scanning .detection-sweep { animation: none; opacity: 0; }
}
</style>
</head>
<body>
<div class="shell">
  <header>
    <div class="brand">
      <div class="brand-mark">FIT</div>
      <div><h1>DRIVER FIT LAB</h1><div class="sub">Morphology summary and seating configuration preview</div></div>
    </div>
    <div class="top-right"><span class="prototype">PROTOTYPE · NO SEAT ACTUATION</span></div>
  </header>

  <main class="layout">
    <div class="card analysis-card">
      <h2 class="step">LOAD, SCAN AND ANALYZE IMAGE</h2>
      <div class="upload-grid">
        <div class="upload-view">
          <label for="frontalFile">Frontal image</label>
          <input id="frontalFile" type="file" accept="image/jpeg,image/png,image/webp">
          <canvas id="frontalCanvas" aria-label="Frontal image calibration preview"></canvas>
          <small>Stand upright with your full body visible. Click the two ends of the vertical reference marker.</small>
        </div>
        <div class="detection-panel">
          <div class="detection-title">Detecting Person</div>
          <div id="detectionViewport" class="detection-viewport" aria-label="Simulated person detection animation">
            <div id="detectionStage" class="detection-stage">
              <img id="detectionImage" class="detection-image" alt="">
              <img id="detectionColorImage" class="detection-image detection-color-image" alt="">
              <svg class="detection-overlay" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
                <g class="camera-links">
                  <line id="cameraToTopLeft"/>
                  <line id="cameraToTopRight"/>
                  <line id="cameraToBottomRight"/>
                  <line id="cameraToBottomLeft"/>
                </g>
                <rect id="detectedPersonBox" class="person-box"/>
                <circle cx="8" cy="90" r="2.6"/>
                <rect class="camera-body" x="4" y="84" width="8" height="4"/>
              </svg>
              <div class="detection-sweep" aria-hidden="true"></div>
            </div>
          </div>
          <div id="detectionStatus" class="detection-status" role="status">Upload an image, then press ANALYZE IMAGE to detect the person.</div>
        </div>
      </div>
      <div class="calibration">
        <label for="referenceLength">Reference length
          <input id="referenceLength" type="number" min="10" max="250" step="0.1" value="100" aria-label="Reference marker length in centimetres">
        </label>
        <small>Marker must be vertical and in the same plane as you. The image is processed locally and not saved.</small>
      </div>
      <button id="analyzeImages" class="upload-submit" type="button">ANALYZE IMAGE</button>
      <div id="profileStatus" class="upload-status" role="status">No images analyzed yet.</div>
    </div>
    <section class="left">
      <div class="card">
        <h2 class="step">DRIVER DIMENSIONS <span class="info" title="Height is estimated from the calibrated full-body detection box.">i</span></h2>
        <div class="height-line"><output id="heightValue">178</output><span>cm estimated total height</span></div>
        <div class="range-wrap"><input id="height" type="range" min="120" max="230" value="178" aria-label="Estimated driver height in centimetres">
          <div class="range-labels"><span>120</span><span>147</span><span>175</span><span>202</span><span>230 cm</span></div></div>
      </div>

      <div class="card">
        <h2 class="step">SIZE CLASS <span class="info" title="Illustrative height bands. Replace with validated population and vehicle-specific calibration.">i</span></h2>
        <div class="profiles" role="group" aria-label="Driver size profile">
          <button class="choice" data-size="small"><span class="person-icon">S</span><strong>Small</strong><small>≤165 cm</small></button>
          <button class="choice active" data-size="medium"><span class="person-icon">M</span><strong>Medium</strong><small>166–185 cm</small></button>
          <button class="choice" data-size="large"><span class="person-icon">L</span><strong>Large</strong><small>≥186 cm</small></button>
        </div>
        <h2 class="step" style="margin-top:17px">BUILD CATEGORY</h2>
        <div class="body-types" role="group" aria-label="Body morphology profile">
          <button class="choice" data-build="athletic"><span class="person-icon">◇</span><strong>Athletic</strong></button>
          <button class="choice active" data-build="average"><span class="person-icon">◯</span><strong>Average</strong></button>
          <button class="choice" data-build="robust"><span class="person-icon">⬭</span><strong>Robust</strong></button>
        </div>
      </div>

    </section>

    <section class="right">
      <div class="card">
        <div class="result-title">APPROXIMATE DRIVER MEASUREMENTS</div>
        <div class="parameters">
          <div class="parameter"><span class="param-icon">↕</span><span class="param-name">Estimated height</span><span class="param-value"><span id="measureHeight">—</span><small>cm</small></span></div>
          <div class="parameter"><span class="param-icon">↔</span><span class="param-name">Shoulder width</span><span class="param-value"><span id="measureShoulder">—</span><small>cm</small></span></div>
          <div class="parameter"><span class="param-icon">↔</span><span class="param-name">Hip width</span><span class="param-value"><span id="measureHip">—</span><small>cm</small></span></div>
          <div class="parameter"><span class="param-icon">⌂</span><span class="param-name">Face height</span><span class="param-value"><span id="measureFace">—</span><small>cm · approximate</small></span></div>
          <div class="parameter"><span class="param-icon">↕</span><span class="param-name">Torso length</span><span class="param-value"><span id="measureTorso">—</span><small>cm</small></span></div>
          <div class="parameter"><span class="param-icon">↕</span><span class="param-name">Leg length</span><span class="param-value"><span id="measureLegs">—</span><small>cm · upper + lower leg</small></span></div>
        </div>
        <div id="measurementNote" class="under-cta">Height uses the frontal full-body detection box scaled by the reference marker. Keep the top of the head and the soles visible. Torso and leg measurements require a side-view image and are unavailable in this one-image workflow.</div>
      </div>
      <div class="card">
        <div class="result-title">MODEL PERFORMANCE</div>
        <div class="parameters">
          <div class="parameter">
            <span class="param-icon">◎</span>
            <span class="param-name">Morphology estimation confidence</span>
            <span class="param-value"><span id="morphologyConfidence">—</span><small>pose keypoint confidence</small></span>
          </div>
          <div class="parameter">
            <span class="param-icon">◎</span>
            <span class="param-name">Face detection confidence</span>
            <span class="param-value"><span id="faceConfidence">—</span><small>YuNet face score</small></span>
          </div>
        </div>
        <div class="under-cta">Model confidence scores indicate detection certainty; they are not measurement-accuracy guarantees.</div>
      </div>
      <div class="card">
        <div class="result-title">CONFIGURATION VECTOR · SIMULATION</div>
        <div class="movement-figures" aria-label="Diagrams of simulated seat movements">
            <div class="movement-row">
              <svg viewBox="0 0 314 300" role="img" aria-label="Seated driver illustration showing backrest recline movement">
                <image class="movement-art" href="/assets/driver-seat-profile.png" x="0" y="0" width="314" height="300"/>
                <path id="backrestMovementArrow" class="movement-arrow" d="M250 269 A28 28 0 0 1 278 241 M271 242 L278 241 L276 248"/>
              </svg>
              <span class="movement-label">Backrest Inclination</span>
              <output class="movement-copy" id="movementBackrestValue" aria-label="Backrest Inclination"><span class="movement-number" id="backrest">100</span><span class="movement-unit">°</span></output>
            </div>
            <div class="movement-row">
              <svg viewBox="0 0 314 300" role="img" aria-label="Seated driver illustration showing vertical seat movement">
                <image class="movement-art" href="/assets/driver-seat-profile.png" x="0" y="0" width="314" height="300"/>
                <path id="verticalMovementArrow" class="movement-arrow" d="M290 243 V197 M284 205 L290 197 L296 205"/>
              </svg>
              <span class="movement-label">Seat Height</span>
              <output class="movement-copy" id="movementVerticalValue" aria-label="Seat Height"><span class="movement-number" id="seatHeight">+20</span><span class="movement-unit">mm</span></output>
            </div>
            <div class="movement-row">
              <svg viewBox="0 0 314 300" role="img" aria-label="Seated driver illustration showing fore and aft seat movement">
                <image class="movement-art" href="/assets/driver-seat-profile.png" x="0" y="0" width="314" height="300"/>
                <path id="horizontalMovementArrow" class="movement-arrow" d="M88 278 H184 M176 272 L184 278 L176 284"/>
              </svg>
              <span class="movement-label">Seat Depth</span>
              <output class="movement-copy" id="movementHorizontalValue" aria-label="Seat Depth"><span class="movement-number" id="seatDepth">+15</span><span class="movement-unit">mm</span></output>
            </div>
        </div>
      </div>
      <div class="card">
        <div class="result-title">FIT SUMMARY</div>
        <div style="color:var(--muted);font-size:11px">Values update from the selected demo profile, height and body build. The orange overlay previews the illustrative seat adjustment against the fixed reference position; no real seat is actuated.</div>
        <button id="preview" class="cta" type="button">PREVIEW SIMULATED ADJUSTMENT</button>
        <div class="under-cta">Click again at any time to replay from the fixed reference position.</div>
        <div class="seat-scheme">
          <img src="/assets/automotive-seat-adjustment-scheme.png" alt="Seat adjustment scheme with a fixed reference seat and three smaller diagrams showing fore-aft, height, and backrest movement.">
          <img id="seatProposal" class="seat-proposal" src="/assets/automotive-seat-proposed-overlay.png" alt="" aria-hidden="true">
          <img id="seatBackrestProposal" class="seat-proposal" src="/assets/automotive-seat-proposed-overlay.png" alt="" aria-hidden="true">
        </div>
        <div class="seat-scheme-legend">
          <span><i style="background:#424d5d"></i>Fixed original position</span>
          <span><i style="background:#f28c28"></i>Simulated seat position</span>
        </div>
        <div id="seatAnimationStatus" class="seat-scheme-status" role="status" aria-live="polite"></div>
      </div>
    </section>
  </main>
  <footer class="foot">
    <span><strong>DEMO ONLY:</strong> parameter offsets are placeholders, not ergonomic or safety recommendations.</span>
    <span>Requires vehicle-specific calibration before real use.</span>
  </footer>
</div>
<script>
(() => {
  const height = document.getElementById("height");
  const sizeButtons = [...document.querySelectorAll("[data-size]")];
  const buildButtons = [...document.querySelectorAll("[data-build]")];
  const references = {small: 158, medium: 178, large: 195};
  const base = {
    small: {seatHeight: 0, seatDepth: 0, backrest: 100},
    medium: {seatHeight: 20, seatDepth: 15, backrest: 100},
    large: {seatHeight: 38, seatDepth: 28, backrest: 102}
  };
  let selectedSize = "medium";
  let selectedBuild = "average";

  const clamp = (n, low, high) => Math.max(low, Math.min(high, n));
  const band = value => value < 166 ? "small" : value <= 185 ? "medium" : "large";

  function selectSize(size, updateHeight) {
    selectedSize = size;
    sizeButtons.forEach(button => button.classList.toggle("active", button.dataset.size === size));
    if (updateHeight) height.value = references[size];
    render();
  }
  function selectBuild(build) {
    selectedBuild = build;
    buildButtons.forEach(button => button.classList.toggle("active", button.dataset.build === build));
    render();
  }
  function render() {
    const h = Number(height.value);
    document.getElementById("heightValue").value = h;
    const suggested = band(h);
    if (suggested !== selectedSize) {
      selectedSize = suggested;
      sizeButtons.forEach(button => button.classList.toggle("active", button.dataset.size === selectedSize));
    }

    const template = base[selectedSize];
    const heightDelta = h - references[selectedSize];
    const seatHeight = clamp(Math.round(template.seatHeight + heightDelta * 0.6), -15, 50);
    const seatDepth = template.seatDepth;
    const backrest = template.backrest;
    const signed = n => n > 0 ? `+${n}` : `${n}`;
    document.getElementById("backrest").textContent = backrest;
    document.getElementById("seatHeight").textContent = signed(seatHeight);
    document.getElementById("seatDepth").textContent = signed(seatDepth);
    document.getElementById("backrestMovementArrow").setAttribute(
      "d",
      backrest < 100
        ? "M278 241 A28 28 0 0 0 250 269 M256 269 L250 269 L252 262"
        : "M250 269 A28 28 0 0 1 278 241 M271 242 L278 241 L276 248"
    );
    document.getElementById("verticalMovementArrow").setAttribute(
      "d",
      seatHeight < 0
        ? "M290 197 V243 M284 235 L290 243 L296 235"
        : "M290 243 V197 M284 205 L290 197 L296 205"
    );
    document.getElementById("horizontalMovementArrow").setAttribute(
      "d",
      seatDepth < 0
        ? "M184 278 H88 M96 272 L88 278 L96 284"
        : "M88 278 H184 M176 272 L184 278 L176 284"
    );
  }

  height.addEventListener("input", () => selectSize(band(Number(height.value)), false));
  sizeButtons.forEach(button => button.addEventListener("click", () => selectSize(button.dataset.size, true)));
  buildButtons.forEach(button => button.addEventListener("click", () => selectBuild(button.dataset.build)));
  const imageView = {
    file: document.getElementById("frontalFile"),
    canvas: document.getElementById("frontalCanvas"),
    image: null,
    points: [],
    dataUrl: null
  };
  const detectionViewport = document.getElementById("detectionViewport");
  const detectionStage = document.getElementById("detectionStage");
  const detectionImage = document.getElementById("detectionImage");
  const detectionColorImage = document.getElementById("detectionColorImage");
  const detectionStatus = document.getElementById("detectionStatus");
  let detectionStatusTimers = [];
  const status = document.getElementById("profileStatus");
  const fmtCm = value => Number.isFinite(value) ? value.toFixed(1) : "n/a";
  const previewButton = document.getElementById("preview");
  const seatProposal = document.getElementById("seatProposal");
  const seatBackrestProposal = document.getElementById("seatBackrestProposal");
  function clearDetectionStatusTimers() {
    detectionStatusTimers.forEach(timer => window.clearTimeout(timer));
    detectionStatusTimers = [];
  }
  function sizeDetectionStage() {
    const aspectRatio = imageView.image.naturalWidth / imageView.image.naturalHeight;
    detectionStage.style.width =
      `${Math.min(detectionViewport.clientWidth, 240 * aspectRatio)}px`;
    detectionStage.style.aspectRatio =
      `${imageView.image.naturalWidth} / ${imageView.image.naturalHeight}`;
  }
  function prepareDetectionView() {
    clearDetectionStatusTimers();
    detectionViewport.classList.remove("scanning");
    detectionViewport.classList.remove("detected");
    sizeDetectionStage();
    detectionColorImage.style.clipPath = "";
    ["x", "y", "width", "height"].forEach(attribute =>
      document.getElementById("detectedPersonBox").removeAttribute(attribute)
    );
    ["cameraToTopLeft", "cameraToTopRight", "cameraToBottomRight", "cameraToBottomLeft"]
      .forEach(id => {
        const line = document.getElementById(id);
        ["x1", "y1", "x2", "y2"].forEach(attribute => line.removeAttribute(attribute));
      });
    detectionStatus.textContent = "Image ready. Press ANALYZE IMAGE to detect the person.";
  }
  function startDetectionAnimation() {
    clearDetectionStatusTimers();
    sizeDetectionStage();
    detectionViewport.classList.remove("detected", "scanning");
    void detectionViewport.offsetWidth;
    detectionViewport.classList.add("scanning");
    detectionStatus.textContent = "Scanning image for a person…";
    const duration = window.matchMedia("(prefers-reduced-motion: reduce)").matches
      ? 0
      : 5000;
    return new Promise(resolve => {
      detectionStatusTimers.push(window.setTimeout(() => {
        detectionStatus.textContent = "Scan complete. Finalizing person detection…";
        resolve();
      }, duration));
      detectionStatusTimers.push(window.setTimeout(() => {
      detectionStatus.textContent = "Detecting person in the frontal image…";
      }, 850));
    });
  }
  function showDetectionBox(box) {
    clearDetectionStatusTimers();
    if (!box || !["x", "y", "width", "height"].every(
      key => Number.isFinite(box[key])
    ) || box.width <= 0 || box.height <= 0) {
      throw new Error("The person detector returned an invalid bounding box.");
    }
    const x = box.x;
    const y = box.y;
    const right = x + box.width;
    const bottom = y + box.height;
    detectionColorImage.style.clipPath =
      `inset(${y}% ${100 - right}% ${100 - bottom}% ${x}%)`;
    const cameraX = 8;
    const cameraY = 90;
    document.getElementById("detectedPersonBox").setAttribute(
      "x", x.toFixed(3)
    );
    document.getElementById("detectedPersonBox").setAttribute(
      "y", y.toFixed(3)
    );
    document.getElementById("detectedPersonBox").setAttribute(
      "width", box.width.toFixed(3)
    );
    document.getElementById("detectedPersonBox").setAttribute(
      "height", box.height.toFixed(3)
    );
    [
      ["cameraToTopLeft", x, y],
      ["cameraToTopRight", right, y],
      ["cameraToBottomRight", right, bottom],
      ["cameraToBottomLeft", x, bottom]
    ].forEach(([id, cornerX, cornerY]) => {
      const line = document.getElementById(id);
      line.setAttribute("x1", cameraX);
      line.setAttribute("y1", cameraY);
      line.setAttribute("x2", cornerX);
      line.setAttribute("y2", cornerY);
    });
    const sweep = detectionStage.querySelector(".detection-sweep");
    sweep.style.setProperty(
      "--scan-distance",
      `${detectionStage.getBoundingClientRect().height}px`
    );
    detectionViewport.classList.remove("scanning");
    detectionViewport.classList.add("detected");
    void detectionViewport.offsetWidth;
    detectionStatus.textContent = "Person detected. Bounding box shown in orange.";
  }
  previewButton.addEventListener("click", () => {
    seatProposal.getAnimations().forEach(animation => animation.cancel());
    seatBackrestProposal.getAnimations().forEach(animation => animation.cancel());
    [seatProposal, seatBackrestProposal].forEach(proposal => {
      proposal.style.opacity = "0";
      proposal.style.transform = "";
    });

    const heightOffset = Number(document.getElementById("seatHeight").textContent);
    const depthOffset = Number(document.getElementById("seatDepth").textContent);
    const backrestAngle = Number(document.getElementById("backrest").textContent);
    const imageWidth = document.querySelector(".seat-scheme").getBoundingClientRect().width;
    const scale = imageWidth / 520;
    const motionScale = 1.5 * 1.5 * 2;
    const horizontal = clamp(-depthOffset * 0.3, -18, 18) * motionScale * scale;
    const vertical = clamp(-heightOffset * 0.35, -18, 18) * motionScale * scale;
    const recline = clamp(backrestAngle - 100, -8, 10);
    const duration = window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : 1250;
    seatProposal.animate(
      [
        {opacity: 0, transform: "translate(0px, 0px)"},
        {opacity: 0.88, transform: `translate(${horizontal}px, ${vertical}px)`}
      ],
      {duration, easing: "cubic-bezier(.2,.8,.2,1)", fill: "forwards"}
    );
    seatBackrestProposal.animate(
      [
        {opacity: 0, transform: "translate(0px, 0px) rotate(0deg)"},
        {opacity: 0.88, transform: `translate(${horizontal}px, ${vertical}px) rotate(${recline}deg)`}
      ],
      {duration, easing: "cubic-bezier(.2,.8,.2,1)", fill: "forwards"}
    );
    document.getElementById("seatAnimationStatus").textContent =
      `Showing orange simulated seat position: fore/aft ${depthOffset >= 0 ? "+" : ""}${depthOffset} mm, height ${heightOffset >= 0 ? "+" : ""}${heightOffset} mm, backrest ${backrestAngle}° (${recline >= 0 ? "+" : ""}${recline}° recline from reference). No real seat movement occurred.`;
  });
  {
    const view = imageView;
    view.file.addEventListener("change", () => {
      const file = view.file.files[0];
      if (!file) return;
      if (file.size > 8 * 1024 * 1024 || !["image/jpeg", "image/png", "image/webp"].includes(file.type)) {
        status.textContent = "Choose a JPEG, PNG, or WebP image smaller than 8 MB.";
        view.file.value = "";
        return;
      }
      const reader = new FileReader();
      reader.onload = () => {
        const image = new Image();
        image.onload = () => {
          view.image = image;
          view.dataUrl = reader.result;
          view.points = [];
          const canvas = view.canvas;
          const scale = Math.min(1, 1200 / image.naturalWidth, 1200 / image.naturalHeight);
          canvas.width = Math.round(image.naturalWidth * scale);
          canvas.height = Math.round(image.naturalHeight * scale);
          canvas.dataset.scale = String(scale);
          canvas.getContext("2d").drawImage(image, 0, 0, canvas.width, canvas.height);
          detectionImage.src = view.dataUrl;
          detectionColorImage.src = view.dataUrl;
          prepareDetectionView();
          status.textContent = "Frontal image loaded. Click the marker's two endpoints.";
        };
        image.onerror = () => { status.textContent = "Could not decode the frontal image."; };
        image.src = reader.result;
      };
      reader.readAsDataURL(file);
    });
    view.canvas.addEventListener("click", event => {
      if (!view.image || view.points.length >= 2) return;
      const rect = view.canvas.getBoundingClientRect();
      const x = (event.clientX - rect.left) * view.canvas.width / rect.width;
      const y = (event.clientY - rect.top) * view.canvas.height / rect.height;
      const scale = Number(view.canvas.dataset.scale);
      view.points.push({x: x / scale, y: y / scale});
      const context = view.canvas.getContext("2d");
      context.drawImage(view.image, 0, 0, view.canvas.width, view.canvas.height);
      context.fillStyle = "#b7472a";
      context.strokeStyle = "#b7472a";
      context.lineWidth = Math.max(2, view.canvas.width / 300);
      view.points.forEach(point => {
        const px = point.x * scale, py = point.y * scale;
        context.beginPath(); context.arc(px, py, context.lineWidth * 2, 0, 2 * Math.PI); context.fill();
      });
      if (view.points.length === 2) {
        context.beginPath();
        context.moveTo(view.points[0].x * scale, view.points[0].y * scale);
        context.lineTo(view.points[1].x * scale, view.points[1].y * scale);
        context.stroke();
      }
    });
  }
  document.getElementById("analyzeImages").addEventListener("click", async event => {
    const button = event.currentTarget;
    const reference = Number(document.getElementById("referenceLength").value);
    if (!imageView.dataUrl || imageView.points.length !== 2) {
      status.textContent = "Upload a frontal image and click both endpoints of the reference marker.";
      return;
    }
    if (!Number.isFinite(reference) || reference < 10 || reference > 250) {
      status.textContent = "Enter a reference length between 10 and 250 cm.";
      return;
    }
    button.disabled = true;
    status.textContent = "Running local pose and morphology analysis…";
    const scanAnimation = startDetectionAnimation();
    try {
      const response = await fetch("/analyze", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          reference_length_cm: reference,
          frontal: {image: imageView.dataUrl, marker_points: imageView.points}
        })
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || "Image analysis failed.");
      await scanAnimation;
      showDetectionBox(result.detection_bbox);
      const h = result.height_cm;
      height.value = String(clamp(Math.round(h), Number(height.min), Number(height.max)));
      selectSize(result.size_class, false);
      selectBuild(result.build_category);
      render();
      const dims = result.dimensions_cm;
      document.getElementById("measureHeight").textContent = h.toFixed(1);
      document.getElementById("measureShoulder").textContent = fmtCm(dims.shoulder_width);
      document.getElementById("measureHip").textContent = fmtCm(dims.hip_width);
      document.getElementById("measureFace").textContent = fmtCm(dims.face_height);
      document.getElementById("measureTorso").textContent = fmtCm(dims.torso);
      document.getElementById("measureLegs").textContent = fmtCm(dims.lower_body);
      document.getElementById("morphologyConfidence").textContent =
        `${(result.confidence * 100).toFixed(0)}%`;
      document.getElementById("faceConfidence").textContent =
        result.face_detection_confidence == null
          ? "Unavailable"
          : `${(result.face_detection_confidence * 100).toFixed(0)}%`;
      document.getElementById("measurementNote").textContent =
        `Estimated height = calibrated full-body detection box: ${h.toFixed(1)} cm. Torso and leg measurements are unavailable from a frontal-only image. ${result.warnings.join(" ")}`;
      status.textContent = "Analysis complete. These are approximate image-based estimates, not ergonomic recommendations.";
    } catch (error) {
      detectionViewport.classList.remove("scanning", "detected");
      clearDetectionStatusTimers();
      detectionStatus.textContent = error.message || "Person detection failed.";
      status.textContent = error.message || "Image analysis failed.";
    } finally {
      button.disabled = false;
    }
  });
  render();
})();
</script>
</body>
</html>
"""


class _HmiHandler(BaseHTTPRequestHandler):
    page: ClassVar[str] = HTML
    pose_estimator: ClassVar[DriverPoseEstimator | None] = None
    face_detector: ClassVar[FaceHeightDetector | None] = None

    def do_GET(self) -> None:
        image_assets = {
            "/assets/driver-seat-profile.png": "driver-seat-profile.png",
            "/assets/automotive-seat-adjustment-scheme.png": (
                "automotive-seat-adjustment-scheme.png"
            ),
            "/assets/automotive-seat-proposed-overlay.png": (
                "automotive-seat-proposed-overlay.png"
            ),
        }
        if self.path in image_assets:
            asset_path = Path(__file__).parent / "assets" / image_assets[self.path]
            try:
                body = asset_path.read_bytes()
            except OSError:
                logger.exception(
                    "Could not read HMI illustration asset: %s", asset_path
                )
                self.send_error(500, "HMI illustration asset unavailable")
                return

            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path not in ("/", "/index.html"):
            self.send_error(404)
            return
        body = self.page.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if self.path != "/analyze":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid Content-Length."})
            return
        if content_length <= 0 or content_length > MAX_REQUEST_BYTES:
            self._send_json(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                {"error": "Request must be smaller than 24 MB."},
            )
            return
        if self.headers.get_content_type() != "application/json":
            self._send_json(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                {"error": "Send the image as application/json."},
            )
            return
        try:
            payload = json.loads(self.rfile.read(content_length))
            result = self._analyze_payload(payload)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError, TypeError) as error:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return
        except Exception:
            logger.exception("Driver morphology inference failed")
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {
                    "error": (
                        "Model inference failed. Check the local YOLO and YuNet "
                        "model setup and logs."
                    )
                },
            )
            return
        self._send_json(HTTPStatus.OK, result)

    def _analyze_payload(self, payload: object) -> dict:
        if not isinstance(payload, dict):
            raise TypeError("Request body must be a JSON object.")
        reference_cm = payload.get("reference_length_cm")
        if not isinstance(reference_cm, (int, float)) or not 10 <= reference_cm <= 250:
            raise ValueError("Reference length must be between 10 and 250 cm.")

        view_payload = payload.get("frontal")
        if not isinstance(view_payload, dict):
            raise TypeError("Missing frontal image and marker calibration.")
        image = self._decode_image(view_payload.get("image"), "frontal")
        marker_pixels = self._marker_length(
            view_payload.get("marker_points"),
            "frontal",
            image.shape[1],
            image.shape[0],
        )

        if self.pose_estimator is None:
            self.__class__.pose_estimator = DriverPoseEstimator(Settings())
        estimator = cast(DriverPoseEstimator, self.pose_estimator)
        if self.face_detector is None:
            self.__class__.face_detector = FaceHeightDetector()
        face_detector = cast(FaceHeightDetector, self.face_detector)
        analyzer = MorphologyAnalyzer(Settings().keypoint_threshold)
        pose_result: FramePoseResult = estimator.estimate(image, 0, 0.0)
        if pose_result.driver is None:
            raise ValueError("No person pose was detected in the frontal image.")
        try:
            frontal = analyzer.analyze_view(pose_result, "frontal")
        except ValueError as error:
            logger.info(
                "Frontal pose detected but morphology measurements are incomplete: %s",
                error,
            )
            frontal = None
        front_pose: PoseDetection = pose_result.driver
        face_detection = face_detector.detect(image, front_pose)

        frontal_span_cm = (
            frontal.face_to_ankle_span_px * reference_cm / marker_pixels
            if frontal is not None
            else None
        )
        shoulder_width = (
            frontal.shoulder_width_over_span * frontal_span_cm
            if (
                frontal is not None
                and frontal.shoulder_width_over_span is not None
                and frontal_span_cm is not None
            )
            else None
        )
        hip_width = (
            frontal.hip_width_over_span * frontal_span_cm
            if (
                frontal is not None
                and frontal.hip_width_over_span is not None
                and frontal_span_cm is not None
            )
            else None
        )
        face_height_cm = (
            face_detection.height * reference_cm / marker_pixels
            if face_detection is not None
            else None
        )
        box_height_cm = (
            front_pose.bounding_box.height * reference_cm / marker_pixels
        )
        height_cm = box_height_cm
        if not 120 <= height_cm <= 230:
            raise ValueError(
                "Full-body detection-box height is outside 120–230 cm. Check that "
                "the frontal image shows the entire person and the marker is "
                "correctly calibrated."
            )
        width_ratio = front_pose.bounding_box.width / max(
            front_pose.bounding_box.height, 1.0
        )
        build_category = (
            "athletic" if width_ratio < 0.32 else "average" if width_ratio < 0.43 else "robust"
        )
        size_class = "small" if height_cm < 166 else "medium" if height_cm <= 185 else "large"
        image_height, image_width = image.shape[:2]
        box_left = max(0.0, min(float(image_width), front_pose.bounding_box.x1))
        box_top = max(0.0, min(float(image_height), front_pose.bounding_box.y1))
        box_right = max(box_left, min(float(image_width), front_pose.bounding_box.x2))
        box_bottom = max(box_top, min(float(image_height), front_pose.bounding_box.y2))
        warnings = [
            "Height uses the frontal full-body detection box scaled by the reference marker; the box must include the top of the head and the soles.",
            "Build class is a rough bounding-box-width proxy and needs validation.",
            "Reference and person must be upright, coplanar, and similarly distant from the camera.",
            "Torso and leg measurements are unavailable because this workflow uses one frontal image.",
            "Seat offsets remain illustrative only.",
        ]
        if frontal is None:
            warnings.append(
                "The person bounding box was detected, but required pose landmarks "
                "were incomplete; shoulder and hip measurements are unavailable."
            )
        if face_detection is None:
            warnings.append(
                "YuNet did not detect a face; the full-body height remains available, "
                "but face height and face confidence are unavailable."
            )
        return {
            "height_cm": round(height_cm, 1),
            "segment_height_cm": None,
            "size_class": size_class,
            "build_category": build_category,
            "detection_bbox": {
                "x": round(box_left / image_width * 100, 3),
                "y": round(box_top / image_height * 100, 3),
                "width": round((box_right - box_left) / image_width * 100, 3),
                "height": round((box_bottom - box_top) / image_height * 100, 3),
            },
            "confidence": round(
                frontal.keypoint_confidence
                if frontal is not None
                else float(
                    np.mean(
                        [
                            point.confidence
                            for point in front_pose.keypoints
                            if point.confidence >= Settings().keypoint_threshold
                        ]
                        or [0.0]
                    )
                ),
                3,
            ),
            "face_detection_confidence": (
                round(face_detection.confidence, 3)
                if face_detection is not None
                else None
            ),
            "dimensions_cm": {
                "shoulder_width": self._round_optional(shoulder_width),
                "hip_width": self._round_optional(hip_width),
                "face_height": self._round_optional(face_height_cm),
                "upper_body": None,
                "lower_body": None,
                "torso": None,
                "upper_leg": None,
                "lower_leg": None,
            },
            "proportions": {
                "torso": None,
                "upper_leg": None,
                "lower_leg": None,
            },
            "warnings": warnings,
        }

    @staticmethod
    def _decode_image(data_url: object, view_name: str) -> np.ndarray:
        if not isinstance(data_url, str) or "," not in data_url:
            raise ValueError(f"Invalid {view_name} image data.")
        header, encoded = data_url.split(",", 1)
        mime_type = header[5:].split(";", 1)[0].lower() if header.startswith("data:") else ""
        if mime_type not in ALLOWED_IMAGE_MIME_TYPES or ";base64" not in header:
            raise ValueError(f"{view_name.capitalize()} image must be JPEG, PNG, or WebP.")
        try:
            image_bytes = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as error:
            raise ValueError(f"Invalid {view_name} image encoding.") from error
        if not image_bytes or len(image_bytes) > MAX_IMAGE_BYTES:
            raise ValueError(f"{view_name.capitalize()} image must be smaller than 8 MB.")
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not decode the {view_name} image.")
        height, width = image.shape[:2]
        if max(height, width) > MAX_IMAGE_DIMENSION:
            raise ValueError(f"{view_name.capitalize()} image dimensions must not exceed 6000 pixels.")
        return image

    @staticmethod
    def _marker_length(
        points: object,
        view_name: str,
        image_width: int,
        image_height: int,
    ) -> float:
        if not isinstance(points, list) or len(points) != 2:
            raise ValueError(f"Click both endpoints of the {view_name} reference marker.")
        parsed: list[tuple[float, float]] = []
        for point in points:
            if not isinstance(point, dict):
                raise TypeError(f"Invalid {view_name} marker coordinates.")
            x, y = point.get("x"), point.get("y")
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                raise TypeError(f"Invalid {view_name} marker coordinates.")
            coordinate = (float(x), float(y))
            if not (0 <= coordinate[0] < image_width and 0 <= coordinate[1] < image_height):
                raise ValueError(
                    f"Reference marker endpoints in the {view_name} image must be "
                    "inside the image."
                )
            parsed.append(coordinate)
        length = float(np.hypot(parsed[0][0] - parsed[1][0], parsed[0][1] - parsed[1][1]))
        if not 10 <= length <= 10000:
            raise ValueError(f"Reference marker in the {view_name} image is too small or large.")
        return length

    @staticmethod
    def _round_optional(value: float | None) -> float | None:
        return round(value, 1) if value is not None else None

    def _send_json(self, status: HTTPStatus, body: dict) -> None:
        response = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format: str, *args: object) -> None:
        logger.info("HMI server: " + format, *args)


def run() -> None:
    parser = argparse.ArgumentParser(
        description="Launch the local driver seat-settings HMI prototype."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Local web server port (default: 8765)",
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    server = HTTPServer(("127.0.0.1", args.port), _HmiHandler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Seat HMI prototype running at {url}")
    print("Press Ctrl+C to stop. No real seat actuation is connected.")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping seat HMI prototype.")
    finally:
        server.server_close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
