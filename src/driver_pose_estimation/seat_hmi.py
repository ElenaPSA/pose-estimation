from __future__ import annotations

import argparse
import base64
import binascii
import json
import logging
import tempfile
import webbrowser
from collections.abc import Callable
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
  --text: #eaf1fb; --muted: #9aa9bc; --accent: #ffb16d;
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
.brand-mark { width: 43px; height: 43px; border-radius: 10px; background: #ffb16d;
  display: grid; place-items: center; color: #0d1724; font-size: 17px; font-weight: 750; }
h1 { margin: 0; font-size: 17px; letter-spacing: .12em; font-weight: 750; }
.sub { margin-top: 3px; color: var(--muted); font-size: 12px; }
.brand h1, .brand .sub { color: #ffb16d; }
.top-right { color: var(--muted); text-align: right; font-size: 11px; }
.prototype { color: #f2d9bc; border: 1px solid #6a4a2f; border-radius: 5px;
  padding: 7px 10px; white-space: nowrap; background: rgba(118, 91, 66, 0.38); font-weight: 750; letter-spacing: .04em; }
.layout { display: grid; grid-template-columns: minmax(340px, .86fr) minmax(450px, 1.4fr);
  gap: 22px; margin-top: 22px; }
.left, .right { display: grid; gap: 15px; align-content: start; }
.card { border: 1px solid var(--line); background: var(--panel);
  border-radius: 9px; padding: 18px 20px; box-shadow: 0 3px 12px rgba(2, 6, 12, 0.5); }
.step { display: flex; gap: 9px; align-items: center; margin: 0 0 14px;
  font-size: 12px; letter-spacing: .08em; font-weight: 750; color: #ffb16d; }
.info { margin-left: auto; color: var(--muted); border: 1px solid #475d78;
  border-radius: 4px; width: 18px; height: 18px; text-align: center; }
.info:focus-visible, button:focus-visible, input:focus-visible {
  outline: 3px solid #d47734; outline-offset: 3px;
}
.height-line { display: flex; align-items: baseline; gap: 8px; }
.height-line output { color: #ffb16d; font-size: 30px; min-width: 56px; font-weight: 650; }
.height-line span { color: var(--muted); }
input[type=range] { width: 100%; accent-color: var(--accent); cursor: pointer; }
.range-wrap { padding: 0 6px; }
.range-labels { display: flex; justify-content: space-between; padding: 0 4px;
  color: var(--muted); font-size: 10px; }
.profiles, .body-types { display: grid; grid-template-columns: repeat(3, 1fr); gap: 9px; }
.choice { min-height: 88px; border: 1px solid var(--line); border-radius: 7px;
  background: #121f2f; cursor: pointer; padding: 10px 7px; text-align: center;
  transition: .15s ease; }
.choice:hover { border-color: #58779a; background: #18273d; transform: translateY(-1px); }
.choice.active { border: 2px solid var(--accent); background: #35271f; box-shadow: inset 0 -3px 0 var(--accent); }
.person-icon { width: 31px; height: 31px; display: grid; place-items: center;
  margin: 0 auto 4px; border-radius: 50%; color: #20150c; background: var(--accent);
  font-size: 12px; line-height: 1; font-weight: 750; }
.choice strong { display: block; font-size: 12px; font-weight: 700; }
.choice small { display: block; color: var(--muted); font-size: 11px; margin-top: 3px; }
.body-types .choice { min-height: 62px; padding-top: 9px; }
.body-types .person-icon { width: 22px; height: 22px; margin-bottom: 3px;
  font-size: 14px; color: #20150c; background: var(--accent); }
.slider-row { margin: 12px 0 3px; }
.slider-head { display: flex; justify-content: space-between; color: var(--muted); font-size: 12px; }
.slider-head output { color: var(--accent); font-weight: 700; }
.slider-wrap { display: grid; grid-template-columns: 48px 1fr 48px; gap: 7px; align-items: center; }
.slider-wrap small { color: var(--muted); font-size: 11px; }
.captured { margin-top: 12px; padding: 11px; border: 1px dashed #465f7e;
  border-radius: 6px; color: var(--text); font-size: 12px; background: var(--panel2); }
.captured input { display: block; width: 100%; margin-top: 7px; color: var(--text); font-size: 12px; }
.upload-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
.upload-view { min-width: 0; padding: 9px; border: 1px solid var(--line); border-radius: 6px; }
.upload-view label { display: block; font-size: 11px; font-weight: 700; }
.upload-view small { color: var(--muted); font-size: 10px; }
.detection-panel { min-width: 0; padding: 9px; border: 1px solid var(--line); border-radius: 6px; background: #09131f; }
.detection-title { color: var(--ink); font-size: 11px; font-weight: 700; }
.detection-viewport { display: flex; justify-content: center; width: 100%; margin-top: 8px; }
.detection-stage { position: relative; flex: none; max-width: 100%; max-height: 300px; overflow: hidden; background: #0b1521; }
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
.upload-submit { margin-top: 12px; padding: 9px 12px; border: 0; border-radius: 5px; background: var(--accent); color: #20150c; font-weight: 700; cursor: pointer; }
.upload-submit:disabled { opacity: .5; cursor: wait; }
.upload-status { margin-top: 9px; font-size: 11px; color: var(--muted); }
.analysis-card { grid-column: 1 / -1; }
.result-title { color: var(--muted); font-size: 11px; letter-spacing: .1em; margin-bottom: 12px; font-weight: 750; }
.result-title.measurements-title { color: #ffb16d; }
.parameters { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.parameter { display: grid; grid-template-columns: 25px 1fr; gap: 7px; align-items: center;
  min-height: 58px; padding: 7px; border: 1px solid var(--line); border-radius: 6px; background: var(--panel2); }
.movement-figures { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; }
.movement-row { min-width: 0; overflow: hidden; border: 1px solid #2c3b50; border-radius: 14px; background: #0b1523; }
.movement-row svg { width: 100%; display: block; }
.movement-label { color: #ffb16d; font-size: 16px; font-weight: 650; }
#seatStudio { padding: clamp(18px, 3vw, 40px); background: linear-gradient(140deg, #142236, #0b1421 55%, #101c2d);
  border: 1px solid #344962; border-radius: 18px; box-shadow: 0 24px 80px #02060e88; }
.studio-heading { display: flex; justify-content: space-between; gap: 20px; align-items: start; }
.studio-heading h2 { font-size: clamp(25px, 3vw, 40px); margin: 8px 0; letter-spacing: -.035em; font-weight: 650; }
.studio-heading p { color: #a9bad0; margin: 0; }
.studio-eyebrow { font-size: 10px; font-weight: 700; letter-spacing: .16em; color: #ffb16d; }
.studio-badge { border: 1px solid #705133; color: #ffc28f; border-radius: 30px; padding: 6px 12px; font-size: 10px; letter-spacing: .09em; white-space: nowrap; }
.studio-reference-strip { display: flex; flex-wrap: wrap; gap: 12px 30px; padding: 18px 0 25px; color: #afc0d5; font-size: 12px; }
.studio-reference-strip b { color: #ecf4ff; margin-right: 7px; }
.studio-reference-strip i { color: #ffb16d; font-style: normal; margin: 0 5px; }
.studio-comfort-reference { padding: 20px; margin-bottom: 24px; border: 1px solid #477b82; border-radius: 12px; background: #10232d; }
.studio-comfort-reference h3 { margin: 8px 0; font-size: 22px; }
.studio-reference-schema { margin: 18px 0; min-width: 0; }
.studio-reference-schema svg { display: block; width: 100%; max-height: 580px; }
@media (max-width: 650px) {
  #seatStudio { min-width: 0; }
  .studio-reference-schema { overflow-x: auto; }
  .studio-reference-schema svg { min-width: 680px; }
}
.studio-comfort-reference .studio-reference-strip { padding: 10px 0; }
.studio-comfort-reference p { margin: 4px 0 0; color: #b0c9d0; font-size: 12px; line-height: 1.7; }
.studio-card-legend { grid-column: 1 / -1; color: #b0becf; font-size: 10px; line-height: 1.6; margin-top: 6px; }
.studio-card-legend i { display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: #899099; margin-right: 5px; vertical-align: middle; }
.studio-card-legend .studio-target-key { background: #ffb16d; margin-left: 8px; }
.studio-card-heading { display: flex; justify-content: space-between; padding: 17px 18px 0; font-size: 10px; letter-spacing: .1em; color: #8fa6c3; }
.studio-direction { color: #6083ad; }
.studio-card-copy { display: grid; grid-template-columns: 1fr auto; padding: 0 20px 20px; gap: 6px; align-items: baseline; }
.studio-card-copy output { color: #ffb16d; font-variant-numeric: tabular-nums; white-space: nowrap; }
.studio-card-copy output b { font-size: 20px; font-weight: 550; letter-spacing: -.04em; }
.studio-card-copy output small { font-size: 13px; color: inherit; margin-left: 5px; }
.studio-range { color: #90a5bf; font-size: 11px; grid-column: 1 / -1; }
.studio-delta { color: #a4c6bf; font-size: 11px; grid-column: 1 / -1; }
.studio-hero { margin-top: 24px; border: 1px solid #31445e; border-radius: 16px; overflow: hidden; background: #0b1523; }
.studio-hero-top { display: flex; justify-content: space-between; align-items: center; gap: 16px; padding: 24px 28px 18px; }
.studio-hero-top h3 { font-size: 22px; font-weight: 550; margin: 5px 0 0; letter-spacing: -.02em; }
.studio-play { cursor: pointer; border: 1px solid #ffb16d; background: linear-gradient(120deg, #ffc38c, #e4873c); color: #20150c; font-weight: 750;
  letter-spacing: .04em; font-size: 11px; padding: 13px 18px; border-radius: 9px; box-shadow: 0 4px 24px #f18e3226; }
.studio-play:hover { background: #ffd2a8; }
.studio-play span { margin-left: 14px; }
.studio-timeline { display: grid; grid-template-columns: repeat(4, 1fr); padding: 0 28px; gap: 10px; color: #7d93ae; font-size: 16px; font-weight: 600; line-height: 1.5; }
.studio-timeline span { padding: 10px 0; border-top: 2px solid #26384e; }
.studio-timeline span.active { border-color: #ffb16d; color: #ffc898; }
.studio-timeline span.complete { border-color: #71ac9e; color: #9acabd; }
.studio-stage { position: relative; }
.studio-stage > svg { display: block; width: 100%; max-height: 610px; }
.studio-stage-label { position: absolute; top: 18px; left: 28px; color: #6e8aa9; font-size: 14px; font-weight: 600; letter-spacing: .12em; }
.studio-stage-label.studio-front { left: auto; right: 28px; }
.studio-live { position: absolute; top: 64px; left: 28px; display: grid; gap: 12px; color: #98afc9; font: 17px "Segoe UI", sans-serif; }
.studio-live b { color: #ffd1a6; font-size: inherit; font-weight: 600; font-variant-numeric: tabular-nums; }
.studio-ghost { opacity: .85; pointer-events: none; filter: grayscale(1); }
.studio-solid { opacity: .88; filter: drop-shadow(0 10px 9px #0008); }
.studio-solid [stroke] { stroke: #ffc38c; }
.studio-solid [fill]:not([fill="none"]):not([fill^="url("]) { fill: #87451f; }
.studio-progress { height: 2px; background: #26364c; }
.studio-progress i { display: block; width: 0; height: 100%; background: #ffb16d; box-shadow: 0 0 12px #f29b55; }
.studio-legend { padding: 15px 28px 8px; display: flex; flex-wrap: wrap; gap: 12px 24px; color: #8fa4bf; font-size: 11px; }
.studio-legend span:last-child { margin-left: auto; font-size: 9px; letter-spacing: .1em; }
.studio-legend i { display: inline-block; background: #53647b; width: 7px; height: 7px; border-radius: 50%; margin-right: 6px; }
.studio-legend span:first-child i { background: #899099; }
.studio-legend span:nth-child(2) i { background: #ffb16d; }
#seatAnimationStatus { padding: 0 28px 20px; color: #a6b8ce; font-size: 12px; min-height: 42px; }
.studio-details { display: grid; grid-template-columns: 1fr 1fr; gap: 36px; margin-top: 28px; padding-top: 26px; border-top: 1px solid #2b3c53; }
.studio-details h3 { font-size: 20px; font-weight: 550; margin: 8px 0; }
.studio-details p { color: #a5b7cd; font-size: 12px; line-height: 1.8; }
.studio-details b { color: #d7e5f7; }
.studio-details .studio-disclaimer { color: #c3a78c; font-size: 11px; }
@media (max-width: 650px) {
  .studio-heading, .studio-hero-top { flex-direction: column; }
  .studio-details { grid-template-columns: 1fr; gap: 15px; }
  .studio-card-copy { padding: 0 15px 18px; }
  .studio-timeline { padding: 0 16px; font-size: 14px; gap: 8px; grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .studio-stage-label { left: 14px; font-size: 12px; }
  .studio-stage-label.studio-front { right: 14px; }
  .studio-live { position: static; display: flex; flex-wrap: wrap; justify-content: space-around; padding: 0 10px 14px; }
  .studio-live > span { white-space: nowrap; }
}
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
      <h2 class="step">LOAD VIDEO / AUTOMATIC FULL-BODY CAPTURE</h2>
      <div class="upload-grid">
        <div class="upload-view">
          <label for="videoFile">Driver video</label>
          <input id="videoFile" type="file" accept="video/*">
          <video id="sourceVideo" hidden muted playsinline preload="metadata" style="width:100%;max-height:300px" aria-label="Uploaded video preview"></video>
          <img id="decodedVideoPreview" hidden alt="Sampled local video preview while full-body frames are checked" style="width:100%;max-height:300px;object-fit:contain">
          <small>A complete near-camera frame is accepted after two consecutive checks of the same person. If no close, complete frame is found, the best stable full-body frame is scanned at the end.</small>
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
          <div id="detectionStatus" class="detection-status" role="status">Upload a video to find a high-confidence full-body frame.</div>
        </div>
      </div>
      <div id="profileStatus" class="upload-status" role="status">No video loaded.</div>
      <p class="under-cta">DEMO ESTIMATES ONLY: scale assumes a 20 cm detected face height. Real height cannot be measured from an uncalibrated video. No vehicle commands.</p>
    </div>
    <section class="left">
      <div class="card">
        <h2 class="step">DRIVER DIMENSIONS <span class="info" title="Uncalibrated demo height assumes the detected face is 20 cm high; not a physical measurement.">i</span></h2>
        <div class="height-line"><output id="heightValue">178</output><span>cm demo height (uncalibrated)</span></div>
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
        <div class="result-title measurements-title">APPROXIMATE DRIVER DEMO ESTIMATES</div>
        <div class="parameters">
          <div class="parameter"><span class="param-icon">↕</span><span class="param-name">Estimated height</span><span class="param-value"><span id="measureHeight">—</span><small>cm</small></span></div>
          <div class="parameter"><span class="param-icon">↔</span><span class="param-name">Shoulder width</span><span class="param-value"><span id="measureShoulder">—</span><small>cm</small></span></div>
          <div class="parameter"><span class="param-icon">↔</span><span class="param-name">Hip width</span><span class="param-value"><span id="measureHip">—</span><small>cm</small></span></div>
          <div class="parameter"><span class="param-icon">⌂</span><span class="param-name">Face height (assumed scale)</span><span class="param-value"><span id="measureFace">—</span><small>cm · not measured</small></span></div>
          <div class="parameter"><span class="param-icon">↕</span><span class="param-name">Torso length</span><span class="param-value"><span id="measureTorso">—</span><small>cm</small></span></div>
          <div class="parameter"><span class="param-icon">↕</span><span class="param-name">Leg length</span><span class="param-value"><span id="measureLegs">—</span><small>cm · upper + lower leg</small></span></div>
        </div>
        <div id="measurementNote" class="under-cta">Uncalibrated demo estimates assume a 20 cm detected face height; they are not physical measurements. Torso and leg lengths are unavailable.</div>
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
    </section>
    <section id="seatStudio" class="card analysis-card" aria-label="Seat configuration vector simulation">
      <noscript>Enable JavaScript to display the seat simulation.</noscript>
    </section>
  </main>
  <footer class="foot">
    <span><strong>DEMO ONLY:</strong> parameter offsets are placeholders, not ergonomic or safety recommendations.</span>
    <span>Requires vehicle-specific calibration before real use.</span>
  </footer>
</div>
<script src="/assets/seat-studio.js"></script>
<script src="/assets/video-capture.js"></script>
<script>
(() => {
  SeatStudio.mount();
  const height = document.getElementById("height");
  const sizeButtons = [...document.querySelectorAll("[data-size]")];
  const buildButtons = [...document.querySelectorAll("[data-build]")];
  const references = {small: 158, medium: 178, large: 195};
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

    SeatStudio.update(h);
  }

  height.addEventListener("input", () => selectSize(band(Number(height.value)), false));
  sizeButtons.forEach(button => button.addEventListener("click", () => selectSize(button.dataset.size, true)));
  buildButtons.forEach(button => button.addEventListener("click", () => selectBuild(button.dataset.build)));
  const imageView = {
    image: null,
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
  function clearDetectionStatusTimers() {
    detectionStatusTimers.forEach(timer => window.clearTimeout(timer));
    detectionStatusTimers = [];
  }
  function sizeDetectionStage() {
    const aspectRatio = imageView.image.naturalWidth / imageView.image.naturalHeight;
    detectionStage.style.width =
      `${Math.min(detectionViewport.clientWidth, 300 * aspectRatio)}px`;
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
    detectionStatus.textContent = "Full-body frame captured. Starting automatic analysis.";
  }
  function startDetectionAnimation(signal) {
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
      const aborted = () => {
        clearDetectionStatusTimers();
        detectionViewport.classList.remove("scanning", "detected");
        resolve();
      };
      signal.addEventListener("abort", aborted, {once: true});
      detectionStatusTimers.push(window.setTimeout(() => {
        signal.removeEventListener("abort", aborted);
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
  VideoCapture.mount({
    async captured(dataUrl, timestamp, signal) {
    const image = new Image();
    image.src = dataUrl;
    await image.decode();
    signal.throwIfAborted();
    imageView.image = image;
    imageView.dataUrl = dataUrl;
    detectionImage.src = dataUrl;
    detectionColorImage.src = dataUrl;
    prepareDetectionView();
    status.textContent = `Analyzing the closest complete frame captured at ${timestamp.toFixed(2)} s…`;
    const scanAnimation = startDetectionAnimation(signal);
    try {
      const response = await fetch("/analyze", {
        method: "POST",
        signal: AbortSignal.any([signal, AbortSignal.timeout(90000)]),
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          mode: "video_demo",
          frontal: {image: imageView.dataUrl}
        })
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || "Image analysis failed.");
      await scanAnimation;
      signal.throwIfAborted();
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
        `Uncalibrated demo height: ${h.toFixed(1)} cm, assuming detected face height = 20 cm. ${result.warnings.join(" ")}`;
      status.textContent = `Automatic analysis complete at ${timestamp.toFixed(2)} s. Demo estimates only, not physical measurements or ergonomic recommendations.`;
    } catch (error) {
      detectionViewport.classList.remove("scanning", "detected");
      clearDetectionStatusTimers();
      detectionStatus.textContent = signal.aborted ? "Video analysis interrupted." : error.message;
      throw error;
    }
    },
    reset() {
      clearDetectionStatusTimers();
      detectionViewport.classList.remove("scanning", "detected");
      detectionImage.removeAttribute("src");
      detectionColorImage.removeAttribute("src");
      detectionStatus.textContent = "Searching video for a high-confidence full-body frame…";
      for (const id of ["measureHeight", "measureShoulder", "measureHip", "measureFace", "measureTorso", "measureLegs", "morphologyConfidence", "faceConfidence"]) {
        document.getElementById(id).textContent = "\u2014";
      }
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
            "/assets/video-capture.js": "video-capture.js",
            "/assets/seat-studio.js": "seat-studio.js",
            "/assets/seat-reference-side.png": "seat-reference-side.png",
            "/assets/seat-reference-definition.png": "seat-reference-definition.png",
            "/assets/seat-reference-ranges.png": "seat-reference-ranges.png",
            "/assets/seat-side-motion.svg": "seat-side-motion.svg",
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
            self.send_header(
                "Content-Type",
                "text/javascript; charset=utf-8"
                if asset_path.suffix == ".js"
                else "image/svg+xml" if asset_path.suffix == ".svg" else "image/png",
            )
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
        if self.path not in ("/analyze", "/probe-frame", "/capture-video", "/stream-video"):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid Content-Length."})
            return
        video_upload = self.path in ("/capture-video", "/stream-video")
        request_limit = 500 * 1024 * 1024 if video_upload else MAX_REQUEST_BYTES
        if content_length <= 0 or content_length > request_limit:
            self._send_json(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                {"error": "Video must be smaller than 500 MB." if video_upload else "Request must be smaller than 24 MB."},
            )
            return
        if not video_upload and self.headers.get_content_type() != "application/json":
            self._send_json(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                {"error": "Send the image as application/json."},
            )
            return
        if self.path == "/stream-video":
            self._stream_video(content_length)
            return
        try:
            if video_upload:
                result = self._capture_video(content_length)
            else:
                payload = json.loads(self.rfile.read(content_length))
                result = (
                    self._probe_frame(payload)
                    if self.path == "/probe-frame"
                    else self._analyze_payload(payload)
                )
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

    def _stream_video(self, content_length: int) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        def emit(event: dict) -> None:
            self.wfile.write(json.dumps(event).encode("utf-8") + b"\n")
            self.wfile.flush()

        try:
            result = self._capture_video(content_length, emit)
            emit({"type": "captured", **result})
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            logger.info("Video preview client disconnected")
        except ValueError as error:
            emit({"error": str(error)})
        except Exception:
            logger.exception("Local video preview failed")
            emit({"error": "Local video preview failed. Check server/model logs."})

    @staticmethod
    def _same_person_track(
        previous: dict[str, float] | None,
        current: dict[str, float] | None,
    ) -> bool:
        if previous is None:
            return current is not None
        if current is None:
            return False

        intersection_width = max(
            0.0,
            min(previous["x2"], current["x2"])
            - max(previous["x1"], current["x1"]),
        )
        intersection_height = max(
            0.0,
            min(previous["y2"], current["y2"])
            - max(previous["y1"], current["y1"]),
        )
        intersection = intersection_width * intersection_height
        previous_area = max(0.0, previous["x2"] - previous["x1"]) * max(
            0.0, previous["y2"] - previous["y1"]
        )
        current_area = max(0.0, current["x2"] - current["x1"]) * max(
            0.0, current["y2"] - current["y1"]
        )
        union = previous_area + current_area - intersection
        overlap = intersection / union if union > 0 else 0.0
        previous_center = (
            (previous["x1"] + previous["x2"]) / 2,
            (previous["y1"] + previous["y2"]) / 2,
        )
        current_center = (
            (current["x1"] + current["x2"]) / 2,
            (current["y1"] + current["y2"]) / 2,
        )
        center_distance = float(
            np.hypot(
                previous_center[0] - current_center[0],
                previous_center[1] - current_center[1],
            )
        )
        return overlap >= 0.05 or center_distance <= 0.08 * np.sqrt(2)

    def _capture_video(
        self, content_length: int, preview: Callable[[dict], None] | None = None
    ) -> dict:
        with tempfile.TemporaryDirectory(prefix="seat-video-") as directory:
            path = Path(directory) / "upload.video"
            remaining = content_length
            with path.open("wb") as stream:
                while remaining:
                    chunk = self.rfile.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError("Video upload was interrupted.")
                    stream.write(chunk)
                    remaining -= len(chunk)
            capture = cv2.VideoCapture(str(path))
            try:
                count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
                fps = capture.get(cv2.CAP_PROP_FPS)
                if not capture.isOpened() or not np.isfinite(count) or not np.isfinite(fps) or count < 1 or fps <= 0:
                    raise ValueError("Cannot decode this video. Try an MP4 or WebM file.")
                samples = min(240, max(1, int(np.ceil(count / fps * 2))), int(count))
                reason = "No suitable person detected."
                confirmations = 0
                tracked_box = None
                recent_observations = []
                best_observation = None
                best_quality = -1.0
                for index in range(samples):
                    frame_number = min(int(count) - 1, int((index + .5) * count / samples))
                    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                    ok, image = capture.read()
                    if not ok:
                        raise ValueError("Could not decode a sampled video frame. Try another video.")
                    h, w = image.shape[:2]
                    scale = min(1.0, 1280 / max(h, w))
                    if scale < 1:
                        image = cv2.resize(image, (round(w * scale), round(h * scale)))
                    data_url = None
                    if preview is not None:
                        ok, encoded = cv2.imencode(".jpg", image)
                        if not ok:
                            raise ValueError("Could not encode the captured video frame.")
                        data_url = "data:image/jpeg;base64," + base64.b64encode(encoded).decode()
                        preview({"type": "frame", "image": data_url, "timestamp": frame_number / fps})
                    result = self._probe_image(image)
                    current_box = result["track_box"]
                    valid_candidate = bool(
                      result["eligible"]
                      and result["confident"]
                      and current_box is not None
                    )
                    same_track = (
                      self._same_person_track(tracked_box, current_box)
                      if tracked_box is not None
                      else valid_candidate
                    )
                    if same_track:
                        tracked_box = current_box
                    eligible = valid_candidate and same_track
                    confirmations = confirmations + 1 if eligible else 0
                    if eligible:
                        if data_url is None:
                            ok, encoded = cv2.imencode(".jpg", image)
                            if not ok:
                                raise ValueError("Could not encode the captured video frame.")
                            data_url = "data:image/jpeg;base64," + base64.b64encode(encoded).decode()
                        observation = {
                            "image": data_url,
                            "timestamp": frame_number / fps,
                            "quality_score": result["quality_score"],
                            "confidence_score": result["confidence_score"],
                          "relative_height": result["relative_height"],
                          "near_camera": result["near_camera"],
                        }
                        recent_observations.append(observation)
                        if len(recent_observations) > 3:
                          recent_observations.pop(0)
                        confirmations = len(recent_observations)
                        if confirmations >= 2 and observation["near_camera"]:
                          return max(
                            recent_observations,
                            key=lambda candidate: candidate["quality_score"],
                          )
                        if confirmations >= 3:
                          for candidate in recent_observations:
                            if candidate["quality_score"] > best_quality:
                              best_observation = candidate
                              best_quality = candidate["quality_score"]
                        reason = (
                          f"Tracking complete body: {confirmations}/3 consecutive frames."
                        )
                    else:
                        recent_observations.clear()
                        confirmations = 0
                        reason = result["reason"] or (
                            "Detection confidence is below the required threshold."
                            if same_track
                            else "The detected person changed."
                        )
                if best_observation is None:
                  raise ValueError(
                    f"No high-confidence complete-body frame in {samples} samples. "
                    f"Required confidence is 82%. {reason} "
                    "Move closer while keeping the head and both feet inside the frame."
                  )
                return best_observation
            finally:
                capture.release()

    @staticmethod
    def _full_body_reason(result: FramePoseResult, image: np.ndarray) -> str | None:
        if result.detected_people != 1 or result.driver is None:
            return "Exactly one person must be visible."
        pose = result.driver
        h, w = image.shape[:2]
        box = pose.bounding_box
        if box.height < 120 or box.width <= 0:
            return "Person is too small for reliable full-body detection."
        if box.x1 < w * .01 or box.y1 < h * .01 or box.x2 > w * .99 or box.y2 > h * .99:
            return "Keep the whole head and feet away from the image edges."
        points = {p.index: p for p in pose.keypoints}
        required = (0, 5, 6, 11, 12, 13, 14, 15, 16)
        if any(
            i not in points
            or points[i].confidence < max(.4, Settings().keypoint_threshold * .8)
            or not (w * .01 <= points[i].x <= w * .99 and h * .01 <= points[i].y <= h * .99)
            for i in required
        ):
            return "Head, shoulders, hips, knees and ankles must all be clearly visible."
        head_gap = points[0].y - box.y1
        ankle_y = max(points[15].y, points[16].y)
        ankle_box_gap = box.y2 - ankle_y
        if not (
            0 <= head_gap <= box.height * .2
            and 0 <= ankle_box_gap <= box.height * .22
            and ankle_y <= h * .93
        ):
          return "Keep the full head and both feet visible with space below the ankles."
        if not (
            points[0].y < min(points[5].y, points[6].y)
            and max(points[5].y, points[6].y) < min(points[11].y, points[12].y)
            and max(points[11].y, points[12].y) < min(points[13].y, points[14].y)
            and max(points[13].y, points[14].y) < min(points[15].y, points[16].y)
            and min(points[15].y, points[16].y) - points[0].y > box.height * .65
        ):
            return "Person must be upright with extended legs, facing the camera."
        return None

    def _probe_frame(self, payload: object) -> dict:
        if not isinstance(payload, dict) or not isinstance(payload.get("frontal"), dict):
            raise TypeError("Missing frontal video frame.")
        image = self._decode_image(payload["frontal"].get("image"), "frontal")
        return self._probe_image(image)

    def _probe_image(self, image: np.ndarray) -> dict:
        if self.pose_estimator is None:
            self.__class__.pose_estimator = DriverPoseEstimator(Settings())
        result = cast(DriverPoseEstimator, self.pose_estimator).estimate(image, 0, 0.0)
        reason = self._full_body_reason(result, image)
        pose = result.driver
        frame_height, frame_width = image.shape[:2]
        track_box = None
        if result.detected_people == 1 and pose is not None:
            box = pose.bounding_box
            track_box = {
                "x1": box.x1 / frame_width,
                "y1": box.y1 / frame_height,
                "x2": box.x2 / frame_width,
                "y2": box.y2 / frame_height,
            }
        face = None
        if reason is None and pose is not None:
            if self.face_detector is None:
                self.__class__.face_detector = FaceHeightDetector()
            face = cast(FaceHeightDetector, self.face_detector).detect(image, pose)
            if face is None or face.height < 12:
                reason = "A clear front-facing face is needed for the uncalibrated demo scale."
            elif not 120 <= pose.bounding_box.height * 20 / face.height <= 230:
                reason = "Face-scale demo estimate is outside 120-230 cm; use a clearer upright frontal view."
        confidence_score = 0.0
        quality_score = 0.0
        relative_height = 0.0
        near_camera = False
        if reason is None and pose is not None and face is not None:
            keypoints = {point.index: point for point in pose.keypoints}
            keypoint_confidence = float(
                np.mean(
                    [
                        keypoints[index].confidence
                        for index in (0, 5, 6, 11, 12, 13, 14, 15, 16)
                    ]
                )
            )
            confidence_product = (
                np.clip(pose.bounding_box.confidence, 0.0, 1.0)
                * np.clip(keypoint_confidence, 0.0, 1.0)
                * np.clip(face.confidence, 0.0, 1.0)
            )
            confidence_score = float(np.cbrt(confidence_product))
            box = pose.bounding_box
            relative_height = min(box.height / max(frame_height, 1), 1.0)
            head_coverage = 1 - (keypoints[0].y - box.y1) / box.height
            ankle_coverage = 1 - (
              box.y2 - max(keypoints[15].y, keypoints[16].y)
            ) / box.height
            completeness = min(head_coverage, ankle_coverage)
            quality_score = (
              relative_height
              * completeness
              * (0.8 + 0.2 * confidence_score)
            )
            near_camera = relative_height >= 0.68
        return {
            "eligible": reason is None,
            "reason": reason,
            "confidence_score": confidence_score,
            "confident": (
                reason is None
                and confidence_score >= 0.70
            ),
            "quality_score": quality_score,
            "relative_height": relative_height,
            "near_camera": near_camera,
            "track_box": track_box,
        }

    def _analyze_payload(self, payload: object) -> dict:
        if not isinstance(payload, dict):
            raise TypeError("Request body must be a JSON object.")
        demo = payload.get("mode") == "video_demo"
        reference_cm = 20.0 if demo else payload.get("reference_length_cm")
        if not isinstance(reference_cm, (int, float)) or not 10 <= reference_cm <= 250:
            raise ValueError("Reference length must be between 10 and 250 cm.")

        view_payload = payload.get("frontal")
        if not isinstance(view_payload, dict):
            raise TypeError("Missing frontal image.")
        image = self._decode_image(view_payload.get("image"), "frontal")
        marker_pixels = 1.0 if demo else self._marker_length(
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
        if demo and (reason := self._full_body_reason(pose_result, image)):
            raise ValueError(reason)
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
        if demo:
            if face_detection is None or face_detection.height < 12:
                raise ValueError("A clear face is required for the face-scale demo estimate.")
            marker_pixels = float(face_detection.height)

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
            if demo:
                raise ValueError(
                    "Uncalibrated face-scale demo height is outside 120-230 cm. "
                    "Try a clearer upright frontal view; no calibrated measurement is available."
                )
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
        if demo:
            warnings = [
                "Uncalibrated demo only: detected face height is assumed to be 20 cm, not measured.",
                "Height = person-box pixels / face-box pixels * 20 cm. All widths inherit this assumed scale.",
                "Perspective, face shape, clothing and pose can strongly bias these estimates.",
                "Build class is a rough bounding-box-width proxy and needs validation.",
                "Torso and leg measurements are unavailable from this single captured frame.",
                "Seat offsets remain illustrative only; not comfort or safety recommendations.",
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
            "estimation_mode": "uncalibrated_face_scale_demo" if demo else "reference_marker",
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
