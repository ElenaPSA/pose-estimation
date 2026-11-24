from __future__ import annotations

import argparse
import base64
import binascii
import json
import logging
import socket
import subprocess
import tempfile
import webbrowser
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import ClassVar, cast

import cv2
import imageio_ffmpeg
import numpy as np

from .camera_calibration import (
  CAMERA_IMAGE_SIZE_PX,
  CAMERA_PITCH_SEARCH_RANGE_DEG,
  CAMERA_ROLL_SEARCH_RANGE_DEG,
  CameraCalibration,
  CameraCalibrationError,
  HeightEstimate,
  SIMULATED_CX_PX,
  SIMULATED_CY_PX,
  SIMULATED_FX_PX,
  SIMULATED_FY_PX,
  SIMULATED_CAMERA_HEIGHT_M,
  SIMULATED_CAMERA_HEIGHT_RANGE_M,
  SIMULATED_CAMERA_PITCH_DEG,
  SIMULATED_CAMERA_ROLL_DEG,
  SIMULATED_CAMERA_YAW_DEG,
  IMPLIED_DIAGONAL_FOV_DEG,
  REPORTED_DIAGONAL_FOV_DEG,
  SIMULATED_HORIZONTAL_FOV_DEG,
  SIMULATED_VERTICAL_FOV_DEG,
)
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
FULL_BODY_CONFIDENCE_THRESHOLD = 0.70

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Driver seat settings prototype</title>
<style>
:root {
  color-scheme: dark;
  --capture-media-height: clamp(220px, calc(100vh - 360px), 650px);
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
.brand h1 { margin: 0; color: #ffb16d; font-size: 15px; letter-spacing: .08em; font-weight: 750; }
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
.profile-card .step { position: relative; justify-content: center; }
.profile-card .info { position: absolute; right: 0; }
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
.detection-viewport[hidden] { display: none; }
.detection-stage { position: relative; flex: none; max-width: 100%; overflow: hidden; background: #0b1521;
  width: min(100%, calc(var(--capture-media-height) * var(--frame-aspect, 1))); }
.detection-image { display: block; width: 100%; height: 100%; object-fit: contain; }
.detection-image[hidden] { display: none; }
.detection-sweep { position: absolute; inset: 0; width: 100%; pointer-events: none; }
.detection-sweep { top: 0; bottom: auto; height: 2px; opacity: 0; background: #ff2638; box-shadow: 0 0 8px 3px rgba(255, 38, 56, .8); }
.detection-viewport.scanning .detection-sweep { animation: person-scan 2s linear infinite; }
@keyframes person-scan {
  0% { top: 0; opacity: 0; }
  8% { opacity: 1; }
  92% { opacity: 1; }
  100% { top: 100%; opacity: .75; }
}
.detection-status { margin-top: 6px; color: var(--muted); font-size: 10px; }
.upload-submit { margin-top: 12px; padding: 9px 12px; border: 0; border-radius: 5px; background: var(--accent); color: #20150c; font-weight: 700; cursor: pointer; }
.upload-submit:disabled { opacity: .5; cursor: wait; }
.upload-status { margin-top: 9px; font-size: 11px; color: var(--muted); }
.analysis-card { grid-column: 1 / -1; }
.result-title { color: var(--muted); font-size: 11px; letter-spacing: .1em; margin-bottom: 12px; font-weight: 750; }
.parameters { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.parameter { display: grid; grid-template-columns: 25px 1fr; gap: 7px; align-items: center;
  min-height: 58px; padding: 7px; border: 1px solid var(--line); border-radius: 6px; background: var(--panel2); }
.morphology-figure { display: grid; grid-template-columns: minmax(0, 1fr) max-content; gap: 14px; align-items: center; margin-top: 16px; padding-top: 14px; border-top: 1px solid var(--line); }
.morphology-figure[hidden] { display: none; }
.morphology-figure-stage { position: relative; justify-self: center; max-width: 100%; overflow: hidden; background: #09131f;
  width: min(100%, calc(var(--capture-media-height) * var(--frame-aspect, 1))); }
.morphology-figure-stage img { display: block; width: 100%; height: 100%; object-fit: contain; }
.morphology-figure-stage svg { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; }
.morphology-figure-stage line { stroke-width: 2; vector-effect: non-scaling-stroke; stroke-linecap: round; }
.morphology-figure-stage circle { stroke: #08111b; stroke-width: .8; vector-effect: non-scaling-stroke; }
.morphology-figure-stage line, .morphology-figure-stage circle { animation: segment-reveal .4s ease both; }
.morphology-figure-stage line:nth-child(3), .morphology-figure-stage circle { animation-delay: .15s; }
.morphology-figure-stage line:nth-child(4), .morphology-figure-stage line:nth-child(5) { animation-delay: .3s; }
.morphology-figure-stage line:nth-child(6), .morphology-figure-stage line:nth-child(7) { animation-delay: .5s; }
.morphology-figure-stage line:nth-child(8), .morphology-figure-stage line:nth-child(9) { animation-delay: .7s; }
.morphology-figure-stage line:nth-child(10), .morphology-figure-stage line:nth-child(11) { animation-delay: .9s; }
@keyframes segment-reveal { from { opacity: 0; } to { opacity: 1; } }
.morphology-shoulders { stroke: var(--accent); }
.morphology-hips { stroke: var(--cyan); }
.morphology-torso { stroke: var(--green); }
.morphology-upper-leg { stroke: var(--coral); }
.morphology-lower-leg { stroke: var(--amber); }
.morphology-upper-arm { stroke: #b6a5ed; }
.morphology-lower-arm { stroke: #8fb8ff; }
.morphology-landmark { fill: var(--text); }
.morphology-figure-legend { display: grid; gap: 8px; }
.morphology-figure-heading { grid-column: 1 / -1; color: var(--muted); font-size: 10px; font-weight: 750; letter-spacing: .08em; }
.morphology-segment { display: grid; grid-template-columns: 12px 84px max-content; justify-content: start; gap: 7px; align-items: center; min-width: 0; font-size: 14px; }
.morphology-segment i { width: 10px; height: 3px; background: var(--segment-color); }
.morphology-segment span { color: var(--muted); }
.morphology-segment output { color: var(--ink); font-size: 15px; font-weight: 700; font-variant-numeric: tabular-nums; white-space: nowrap; }
.morphology-figure-note { grid-column: 1 / -1; color: var(--muted); font-size: 10px; line-height: 1.5; }
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
.studio-card-copy .angle-unit { position: relative; top: -.25em; margin-left: 1px; }
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
.studio-details .studio-reference-schema { grid-row: span 2; margin: 0; }
.studio-details .studio-reference-schema svg { max-height: 580px; }
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
.screen-nav { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
.screen-nav button { border: 1px solid var(--line); background: var(--panel2); border-radius: 6px; padding: 8px 14px; cursor: pointer; }
.screen-nav button[aria-pressed="true"] { border-color: var(--accent); color: var(--accent); background: #35271f; }
.capture-layout { display: grid; grid-template-columns: minmax(0, 2fr) minmax(250px, .7fr); gap: 12px; align-items: start; }
.capture-layout > .left { align-self: center; }
.capture-layout[hidden], [data-studio-view][hidden] { display: none; }
.camera-details { color: var(--muted); font-size: 11px; margin-top: 8px; }
.camera-details summary { cursor: pointer; }
@media (min-width: 900px) and (min-height: 620px) {
  .shell { max-width: none; padding: 12px 20px; }
  header { padding-bottom: 8px; }
  .layout { display: block; margin-top: 10px; }
  .card { padding: 12px; }
  .choice { min-height: 65px; padding: 6px 3px; }
  .body-types .choice { min-height: 52px; padding: 4px; }
  #sourceVideo, #decodedVideoPreview { max-height: var(--capture-media-height); }
  .morphology-figure { margin-top: 6px; padding-top: 6px; gap: 6px; }
  .morphology-figure-legend { gap: 4px; }
  #seatStudio { padding: 14px 18px; }
  #seatStudio[data-active-screen="animation"] .studio-heading,
  #seatStudio[data-active-screen="reference"] .studio-heading { display: none; }
  .studio-heading h2 { font-size: 22px; margin: 3px 0; }
  .studio-heading p { font-size: 11px; }
  .movement-figures { gap: 10px; }
  .movement-row svg { height: min(25vh, 185px); }
  .studio-card-heading { padding: 8px 12px 0; }
  .studio-card-copy { padding: 0 12px 10px; gap: 3px; }
  .studio-hero { margin-top: 8px; }
  .studio-hero-top { padding: 8px 16px; }
  .studio-hero-top h3 { font-size: 17px; }
  .studio-play { padding: 8px 12px; }
  .studio-timeline { padding: 0 16px; font-size: 12px; }
  .studio-timeline span { padding: 4px 0; }
  .studio-stage > svg { height: max(180px, calc(100vh - 385px)); max-height: none; }
  .studio-stage-label { top: 8px; font-size: 11px; }
  .studio-live { top: 38px; gap: 6px; font-size: 13px; }
  .studio-legend { padding: 6px 16px; }
  #seatAnimationStatus { padding: 0 16px 8px; min-height: 0; }
  .studio-details { margin-top: 8px; padding-top: 8px; }
  .studio-details p { line-height: 1.5; }
  .studio-details .studio-reference-schema svg { max-height: max(220px, calc(100vh - 250px)); }
  .foot { margin-top: 8px; padding-top: 6px; }
}
@media (max-width: 899px) { .capture-layout { grid-template-columns: 1fr; } }
@media (max-width: 900px) {
  .layout { grid-template-columns: 1fr; }
}
@media (max-width: 540px) {
  .shell { padding: 15px 12px; }
  .upload-grid { grid-template-columns: 1fr; }
  header { align-items: flex-start; }
  .top-right { display: none; }
  .movement-figures { grid-template-columns: 1fr; }
  .morphology-figure { grid-template-columns: 1fr; }
  .profiles, .body-types { gap: 5px; }
  .choice { padding-left: 3px; padding-right: 3px; }
  .foot { display: block; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { scroll-behavior: auto !important; }
  .detection-viewport.scanning .detection-sweep { animation: none; opacity: 0; }
  .morphology-figure-stage line, .morphology-figure-stage circle { animation: none; }
}
</style>
</head>
<body>
<div class="shell">
  <header>
    <div class="brand">
      <h1>MORPHOLOGY SUMMARY AND SEATING CONFIGURATION PREVIEW</h1>
    </div>
    <div class="top-right"><span class="prototype">PROTOTYPE · NO SEAT ACTUATION</span></div>
  </header>

  <nav class="screen-nav" aria-label="HMI views">
    <button type="button" data-screen="capture" aria-pressed="true">Video &amp; profile</button>
    <button type="button" data-screen="adjustments" aria-pressed="false">Seat adjustments</button>
    <button type="button" data-screen="animation" aria-pressed="false">Seat animation</button>
    <button type="button" data-screen="reference" aria-pressed="false">Reference details</button>
  </nav>
  <main class="layout">
    <div id="captureScreen" class="capture-layout">
    <div class="card">
      <h2 class="step">LOAD VIDEO / AUTOMATIC FULL-BODY CAPTURE</h2>
      <div class="upload-grid">
        <div class="upload-view">
          <label for="videoFile">Driver video</label>
          <input id="videoFile" type="file" accept="video/*">
          <div id="videoFileStatus" class="upload-status">No video selected.</div>
          <video id="sourceVideo" hidden muted playsinline controls preload="metadata" style="width:100%;object-fit:contain" aria-label="Uploaded video preview"></video>
          <img id="decodedVideoPreview" hidden alt="Sampled local video preview while full-body frames are checked" style="width:100%;object-fit:contain">
        </div>
        <div class="detection-panel">
          <div class="detection-title">Detecting Person / selected frame</div>
          <div id="detectionViewport" class="detection-viewport" aria-label="Simulated person detection animation">
            <div id="detectionStage" class="detection-stage">
              <img id="detectionImage" class="detection-image" hidden alt="">
              <canvas id="detectionVideoFrame" class="detection-image" hidden aria-label="Playing video mirrored during person detection"></canvas>
              <div class="detection-sweep" aria-hidden="true"></div>
            </div>
          </div>
          <div id="morphologyFigure" class="morphology-figure" hidden>
            <div class="morphology-figure-heading">DETECTED BODY / ESTIMATED SEGMENTS</div>
            <div id="morphologyFigureStage" class="morphology-figure-stage">
              <img id="morphologyFigureImage" alt="Selected video frame showing detected driver and morphology keypoints">
              <svg viewBox="0 0 100 100" preserveAspectRatio="none" role="img" aria-label="Overlay of detected body landmarks used for segment estimates">
                <g fill="none">
                  <line id="morphologyShoulders" class="morphology-shoulders"/>
                  <line id="morphologyHips" class="morphology-hips"/>
                  <line id="morphologyTorso" class="morphology-torso"/>
                  <line id="morphologyUpperLeft" class="morphology-upper-leg"/>
                  <line id="morphologyUpperRight" class="morphology-upper-leg"/>
                  <line id="morphologyLowerLeft" class="morphology-lower-leg"/>
                  <line id="morphologyLowerRight" class="morphology-lower-leg"/>
                  <line id="morphologyUpperArmLeft" class="morphology-upper-arm"/>
                  <line id="morphologyUpperArmRight" class="morphology-upper-arm"/>
                  <line id="morphologyLowerArmLeft" class="morphology-lower-arm"/>
                  <line id="morphologyLowerArmRight" class="morphology-lower-arm"/>
                </g>
                <g aria-hidden="true">
                  <circle id="morphologyPointLeftShoulder" class="morphology-landmark" r=".65"/>
                  <circle id="morphologyPointRightShoulder" class="morphology-landmark" r=".65"/>
                  <circle id="morphologyPointLeftElbow" class="morphology-landmark" r=".65"/>
                  <circle id="morphologyPointRightElbow" class="morphology-landmark" r=".65"/>
                  <circle id="morphologyPointLeftWrist" class="morphology-landmark" r=".65"/>
                  <circle id="morphologyPointRightWrist" class="morphology-landmark" r=".65"/>
                  <circle id="morphologyPointLeftHip" class="morphology-landmark" r=".65"/>
                  <circle id="morphologyPointRightHip" class="morphology-landmark" r=".65"/>
                  <circle id="morphologyPointLeftKnee" class="morphology-landmark" r=".65"/>
                  <circle id="morphologyPointRightKnee" class="morphology-landmark" r=".65"/>
                  <circle id="morphologyPointLeftAnkle" class="morphology-landmark" r=".65"/>
                  <circle id="morphologyPointRightAnkle" class="morphology-landmark" r=".65"/>
                </g>
              </svg>
            </div>
            <div class="morphology-figure-legend">
              <div class="morphology-segment" style="--segment-color:var(--accent)"><i></i><span>Shoulders</span><output id="figureShoulderLength">—</output></div>
              <div class="morphology-segment" style="--segment-color:var(--cyan)"><i></i><span>Hips</span><output id="figureHipLength">—</output></div>
              <div class="morphology-segment" style="--segment-color:var(--green)"><i></i><span>Torso</span><output id="figureTorsoLength">—</output></div>
              <div class="morphology-segment" style="--segment-color:var(--coral)"><i></i><span>Upper legs</span><output id="figureUpperLegLength">—</output></div>
              <div class="morphology-segment" style="--segment-color:var(--amber)"><i></i><span>Lower legs</span><output id="figureLowerLegLength">—</output></div>
              <div class="morphology-segment" style="--segment-color:#b6a5ed"><i></i><span>Upper arms</span><output id="figureUpperArmLength">—</output></div>
              <div class="morphology-segment" style="--segment-color:#8fb8ff"><i></i><span>Lower arms</span><output id="figureLowerArmLength">—</output></div>
            </div>
            <div class="morphology-figure-note">2D landmarks; lengths are uncalibrated estimates, not independent 3D anatomical measurements.</div>
          </div>
          <div id="detectionStatus" class="detection-status" role="status">Upload a video to find a high-confidence full-body frame.</div>
        </div>
      </div>
    </div>
    <section class="left">
      <div class="card profile-card">
        <h2 class="step">SIZE CLASS <span class="info" title="Illustrative height bands. Replace with validated population and vehicle-specific calibration.">i</span></h2>
        <div class="profiles" role="group" aria-label="Driver size profile">
          <button class="choice" data-size="small"><span class="person-icon">S</span><strong>Small</strong><small>&lt;160 cm</small></button>
          <button class="choice active" data-size="medium"><span class="person-icon">M</span><strong>Medium</strong><small>160–180 cm</small></button>
          <button class="choice" data-size="large"><span class="person-icon">L</span><strong>Large</strong><small>&gt;180 cm</small></button>
        </div>
        <p class="under-cta">Selecting a size manually uses a representative height; a detected torso is required for a backrest estimate.</p>
        <h2 class="step" style="margin-top:17px">BUILD CATEGORY</h2>
        <div class="body-types" role="group" aria-label="Body morphology profile">
          <button class="choice" data-build="athletic"><span class="person-icon">◇</span><strong>Athletic</strong></button>
          <button class="choice active" data-build="average"><span class="person-icon">◯</span><strong>Average</strong></button>
          <button class="choice" data-build="robust"><span class="person-icon">⬭</span><strong>Robust</strong></button>
        </div>
        <div id="profileStatus" class="upload-status" role="status">No video loaded.</div>
        <details class="camera-details"><summary>Camera and frame selection assumptions (not calibrated)</summary><p>A complete near-camera frame is accepted after two consecutive checks of the same person. If none is found, the best stable full-body frame is selected at the end.</p><p>Mirror position: X 74 cm, Y -85 cm, Z 87 cm; camera height above ground: 115 cm. Model pose: yaw +93° from forward toward vehicle-left, effective pitch −11.5°, roll −0.1°; yaw direction is unverified. Pitch and roll were fitted to seven videos of three people; the effective pitch differs from the user-supplied −15°. These are not measured physical angles. Estimated intrinsics: fx 830 px, fy 1078 px, cx 964 px, cy 544 px at 1928 × 1088. Portrait video assumes a 90° anticlockwise rotation and resize to frame dimensions without cropping. Camera pose and lens distortion are not calibrated. No vehicle commands.</p></details>
      </div>

    </section>

    </div>
    <p id="seatModelUnavailable" class="analysis-card" role="status" hidden>Seat movement preview is unavailable outside the supported height input range.</p>
    <section id="seatStudio" class="card analysis-card" aria-label="Seat configuration vector simulation">
      <noscript>Enable JavaScript to display the seat simulation.</noscript>
    </section>
  </main>
  <footer class="foot">
    <span><strong>DEMO ONLY:</strong> estimated dimensions are not calibrated physical measurements.</span>
    <span>No vehicle commands are sent.</span>
  </footer>
</div>
<script src="/assets/seat-studio.js"></script>
<script src="/assets/video-capture.js"></script>
<script>
(() => {
  SeatStudio.mount();
  const sizeButtons = [...document.querySelectorAll("[data-size]")];
  const buildButtons = [...document.querySelectorAll("[data-build]")];
  const representativeHeights = {small: 158, medium: 178, large: 195};
  let seatAvailable = true;
  let activeScreen = "capture";
  function showScreen(screen) {
    activeScreen = screen;
    document.getElementById("captureScreen").hidden = screen !== "capture";
    document.getElementById("seatStudio").dataset.activeScreen = screen;
    document.getElementById("seatStudio").hidden = screen === "capture" || !seatAvailable;
    document.getElementById("seatModelUnavailable").hidden = screen === "capture" || seatAvailable;
    document.querySelectorAll("[data-studio-view]").forEach(panel => {
      panel.hidden = panel.dataset.studioView !== screen;
    });
    document.querySelectorAll("[data-screen]").forEach(button => {
      button.setAttribute("aria-pressed", String(button.dataset.screen === screen));
    });
  }
  document.querySelectorAll("[data-screen]").forEach(button =>
    button.addEventListener("click", () => showScreen(button.dataset.screen))
  );

  function updateSeat(height, torsoCm = null) {
    seatAvailable = SeatStudio.supports(height);
    if (seatAvailable) SeatStudio.update(height, torsoCm);
    showScreen(activeScreen);
  }

  function selectSize(size) {
    sizeButtons.forEach(button => button.classList.toggle("active", button.dataset.size === size));
  }
  function selectBuild(build) {
    buildButtons.forEach(button => button.classList.toggle("active", button.dataset.build === build));
  }

  sizeButtons.forEach(button => button.addEventListener("click", () => {
    selectSize(button.dataset.size);
    updateSeat(representativeHeights[button.dataset.size]);
  }));
  buildButtons.forEach(button => button.addEventListener("click", () => selectBuild(button.dataset.build)));
  updateSeat(representativeHeights.medium);
  const imageView = {
    image: null,
    dataUrl: null
  };
  const detectionViewport = document.getElementById("detectionViewport");
  const detectionStage = document.getElementById("detectionStage");
  const detectionImage = document.getElementById("detectionImage");
  const detectionVideoFrame = document.getElementById("detectionVideoFrame");
  const detectionStatus = document.getElementById("detectionStatus");
  const status = document.getElementById("profileStatus");
  let mirrorVideo = null, mirrorRequest = null, mirrorSeekHandler = null, mirrorUsesVideoFrames = false;
  function stopPlaybackMirror() {
    if (mirrorVideo && mirrorRequest !== null) {
      if (mirrorUsesVideoFrames) mirrorVideo.cancelVideoFrameCallback(mirrorRequest);
      else cancelAnimationFrame(mirrorRequest);
    }
    if (mirrorVideo && mirrorSeekHandler) mirrorVideo.removeEventListener("seeked", mirrorSeekHandler);
    mirrorVideo = null;
    mirrorRequest = null;
    mirrorSeekHandler = null;
  }
  function startPlaybackMirror(video) {
    stopPlaybackMirror();
    const context = detectionVideoFrame.getContext("2d");
    if (!context || !video.videoWidth || !video.videoHeight) {
      throw new Error("Mirrored video preview is unavailable.");
    }
    const scale = Math.min(1, 1280 / Math.max(video.videoWidth, video.videoHeight));
    detectionVideoFrame.width = Math.round(video.videoWidth * scale);
    detectionVideoFrame.height = Math.round(video.videoHeight * scale);
    detectionStage.style.setProperty("--frame-aspect", video.videoWidth / video.videoHeight);
    detectionStage.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
    detectionImage.hidden = true;
    detectionVideoFrame.hidden = false;
    detectionViewport.hidden = false;
    detectionViewport.classList.add("scanning");
    detectionStatus.textContent = "Mirroring video while checking for a complete person…";
    mirrorVideo = video;
    mirrorUsesVideoFrames = typeof video.requestVideoFrameCallback === "function" &&
      typeof video.cancelVideoFrameCallback === "function";
    const draw = () => {
      if (video.readyState >= 2) {
        context.drawImage(video, 0, 0, detectionVideoFrame.width, detectionVideoFrame.height);
      }
    };
    const next = () => {
      if (mirrorVideo !== video) return;
      draw();
      mirrorRequest = mirrorUsesVideoFrames
        ? video.requestVideoFrameCallback(next)
        : requestAnimationFrame(next);
    };
    mirrorSeekHandler = draw;
    video.addEventListener("seeked", mirrorSeekHandler);
    next();
  }
  window.addEventListener("pagehide", stopPlaybackMirror);
  detectionImage.addEventListener("load", () => {
    if (!detectionImage.naturalWidth || !detectionImage.naturalHeight) return;
    detectionStage.style.setProperty("--frame-aspect", detectionImage.naturalWidth / detectionImage.naturalHeight);
    detectionStage.style.aspectRatio = `${detectionImage.naturalWidth} / ${detectionImage.naturalHeight}`;
  });
  const fmtCm = value => Number.isFinite(value) ? value.toFixed(1) : "n/a";
  function renderMorphologyFigure(dataUrl, keypoints, dimensions) {
    const figure = document.getElementById("morphologyFigure");
    const stage = document.getElementById("morphologyFigureStage");
    const image = document.getElementById("morphologyFigureImage");
    const point = index => {
      const location = keypoints?.[index];
      return location && Number.isFinite(location.x) && Number.isFinite(location.y)
        ? location : null;
    };
    const center = (first, second) => first && second
      ? {x: (first.x + second.x) / 2, y: (first.y + second.y) / 2}
      : null;
    const endpoints = {
      morphologyShoulders: [point(5), point(6)],
      morphologyHips: [point(11), point(12)],
      morphologyTorso: [center(point(5), point(6)), center(point(11), point(12))],
      morphologyUpperLeft: [point(11), point(13)],
      morphologyUpperRight: [point(12), point(14)],
      morphologyLowerLeft: [point(13), point(15)],
      morphologyLowerRight: [point(14), point(16)],
      morphologyUpperArmLeft: [point(5), point(7)],
      morphologyUpperArmRight: [point(6), point(8)],
      morphologyLowerArmLeft: [point(7), point(9)],
      morphologyLowerArmRight: [point(8), point(10)],
    };
    for (const [id, pair] of Object.entries(endpoints)) {
      const line = document.getElementById(id);
      const visible = Boolean(pair[0] && pair[1]);
      line.style.display = visible ? "" : "none";
      if (visible) {
        line.setAttribute("x1", pair[0].x);
        line.setAttribute("y1", pair[0].y);
        line.setAttribute("x2", pair[1].x);
        line.setAttribute("y2", pair[1].y);
      }
    }
    for (const [id, index] of Object.entries({
      morphologyPointLeftShoulder: 5,
      morphologyPointRightShoulder: 6,
      morphologyPointLeftElbow: 7,
      morphologyPointRightElbow: 8,
      morphologyPointLeftWrist: 9,
      morphologyPointRightWrist: 10,
      morphologyPointLeftHip: 11,
      morphologyPointRightHip: 12,
      morphologyPointLeftKnee: 13,
      morphologyPointRightKnee: 14,
      morphologyPointLeftAnkle: 15,
      morphologyPointRightAnkle: 16,
    })) {
      const marker = document.getElementById(id);
      const location = point(index);
      marker.style.display = location ? "" : "none";
      if (location) {
        marker.setAttribute("cx", location.x);
        marker.setAttribute("cy", location.y);
      }
    }
    stage.style.setProperty("--frame-aspect", imageView.image.naturalWidth / imageView.image.naturalHeight);
    stage.style.aspectRatio = `${imageView.image.naturalWidth} / ${imageView.image.naturalHeight}`;
    image.src = dataUrl;
    for (const [id, key] of Object.entries({
      figureShoulderLength: "shoulder_width",
      figureHipLength: "hip_width",
      figureTorsoLength: "torso_length",
      figureUpperLegLength: "upper_leg_length",
      figureLowerLegLength: "lower_leg_length",
      figureUpperArmLength: "upper_arm_length",
      figureLowerArmLength: "lower_arm_length",
    })) {
      document.getElementById(id).textContent = `${fmtCm(dimensions?.[key])} cm`;
    }
    figure.hidden = false;
    return [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16].some(point);
  }
  VideoCapture.mount({
    playback(video) {
      startPlaybackMirror(video);
    },
    progress(dataUrl, timestamp) {
      stopPlaybackMirror();
      detectionViewport.hidden = false;
      detectionViewport.classList.add("scanning");
      detectionVideoFrame.hidden = true;
      detectionImage.hidden = false;
      detectionImage.src = dataUrl;
      detectionStatus.textContent = `Checking sampled frame at ${timestamp.toFixed(2)} s…`;
    },
    failed() {
      stopPlaybackMirror();
      detectionViewport.classList.remove("scanning");
    },
    async captured(dataUrl, timestamp, signal, sampledPreview = false) {
    const image = new Image();
    image.src = dataUrl;
    await image.decode();
    signal.throwIfAborted();
    imageView.image = image;
    imageView.dataUrl = dataUrl;
    status.textContent = `Analyzing the closest complete frame captured at ${timestamp.toFixed(2)} s…`;
    detectionStatus.textContent = sampledPreview
      ? `Analyzing sampled frame at ${timestamp.toFixed(2)} s…`
      : `Mirroring video while analyzing selected frame at ${timestamp.toFixed(2)} s…`;
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
      signal.throwIfAborted();
      stopPlaybackMirror();
      detectionViewport.classList.remove("scanning");
      detectionViewport.hidden = true;
      const calibrated = result.calibration_mode === "vehicle_camera";
      const simulated = result.calibration_mode === "simulated_vehicle_camera";
      selectSize(result.size_class);
      selectBuild(result.build_category);
      const morphology = result.morphology_dimensions_cm;
      updateSeat(result.height_cm, morphology?.torso_length);
      const hasLandmarks = renderMorphologyFigure(dataUrl, result.morphology_figure_keypoints, morphology);
      detectionStatus.textContent = `${hasLandmarks ? "2D landmarks shown" : "No reliable pose landmarks available"}. Selected still: ${timestamp.toFixed(2)} s. ${
        sampledPreview ? "Sampled preview only." : "Video plays independently."}`;
      status.textContent = calibrated
        ? `Calibrated camera height estimate complete at ${timestamp.toFixed(2)} s. Review the calibration and geometry residual before use.`
        : simulated
          ? `Simulated camera estimate complete at ${timestamp.toFixed(2)} s. This is not calibrated.`
          : `Reference-marker analysis complete at ${timestamp.toFixed(2)} s.`;
    } catch (error) {
      stopPlaybackMirror();
      detectionViewport.classList.remove("scanning");
      detectionStatus.textContent = signal.aborted ? "Video analysis interrupted." : error.message;
      throw error;
    }
    },
    reset() {
      stopPlaybackMirror();
      detectionViewport.classList.remove("scanning");
      detectionViewport.hidden = false;
      detectionVideoFrame.hidden = true;
      detectionImage.hidden = true;
      detectionImage.removeAttribute("src");
      document.getElementById("morphologyFigure").hidden = true;
      document.getElementById("morphologyFigureImage").removeAttribute("src");
      const selectedSize = sizeButtons.find(button => button.classList.contains("active")).dataset.size;
      updateSeat(representativeHeights[selectedSize]);
      detectionStatus.textContent = "Searching video for a high-confidence full-body frame…";
      for (const id of ["figureShoulderLength", "figureHipLength", "figureTorsoLength", "figureUpperArmLength", "figureLowerArmLength", "figureUpperLegLength", "figureLowerLegLength"]) {
        document.getElementById(id).textContent = "\u2014";
      }
    }
  });
})();
</script>
</body>
</html>
"""


class _HmiHandler(BaseHTTPRequestHandler):
    page: ClassVar[str] = HTML
    script_assets: ClassVar[dict[str, bytes]] = {
        name: (Path(__file__).parent / "assets" / name).read_bytes()
        for name in ("video-capture.js", "seat-studio.js")
    }
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
                body = self.script_assets.get(asset_path.name)
                if body is None:
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
        if self.path not in (
            "/analyze",
            "/probe-frame",
            "/capture-video",
            "/stream-video",
            "/transcode-video",
        ):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid Content-Length."})
            return
        video_upload = self.path in (
            "/capture-video",
            "/stream-video",
            "/transcode-video",
        )
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
        if self.path == "/transcode-video":
            try:
                self._transcode_video(content_length)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                logger.info("Transcoded video preview client disconnected")
            except (ValueError, TypeError) as error:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            except Exception:
                logger.exception("Local browser-compatible transcode failed")
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": "Could not transcode this video for browser playback."},
                )
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

    def _transcode_video(self, content_length: int) -> None:
        with tempfile.TemporaryDirectory(prefix="seat-preview-") as directory:
            source_path = Path(directory) / "source.video"
            output_path = Path(directory) / "preview.mp4"
            remaining = content_length
            with source_path.open("wb") as stream:
                while remaining:
                    chunk = self.rfile.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError("Video upload was interrupted during transcode.")
                    stream.write(chunk)
                    remaining -= len(chunk)

            capture = cv2.VideoCapture(str(source_path))
            try:
                if not capture.isOpened():
                    raise ValueError("Cannot decode this video for browser playback.")
                source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if source_width < 1 or source_height < 1:
                    raise ValueError("Video contains no decodable frames.")
                scale = min(1.0, 640 / max(source_height, source_width))
                output_width = max(
                    2,
                    round(source_width * scale) - round(source_width * scale) % 2,
                )
                output_height = max(
                    2,
                    round(source_height * scale) - round(source_height * scale) % 2,
                )
            finally:
                capture.release()

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            command = [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(source_path),
                "-vf",
                f"scale={output_width}:{output_height},fps=12",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-crf",
                "28",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output_path),
            ]
            try:
                subprocess.run(
                    command,
                    capture_output=True,
                    check=True,
                    text=True,
                    timeout=600,
                )
            except (OSError, subprocess.SubprocessError) as error:
                details = getattr(error, "stderr", "")
                raise ValueError(
                    "Could not convert video to browser-compatible H.264: "
                    f"{str(details or error)[:500]}"
                ) from error

            if not output_path.is_file() or output_path.stat().st_size == 0:
                raise ValueError("Video transcoding produced no playable frames.")
            output_size_bytes = output_path.stat().st_size
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(output_size_bytes))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            with output_path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    self.wfile.write(chunk)

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
                samples = min(240, max(1, int(np.ceil(count / fps * 4))), int(count))
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
                    f"Required confidence is {FULL_BODY_CONFIDENCE_THRESHOLD:.0%}. {reason} "
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
        confidence_score = 0.0
        quality_score = 0.0
        relative_height = 0.0
        near_camera = False
        if reason is None and pose is not None:
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
            )
            confidence_score = float(np.sqrt(confidence_product))
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
            try:
              simulated_camera = CameraCalibration.simulated_for_image(
                (frame_width, frame_height)
              )
              self._estimate_calibrated_height(simulated_camera, pose, image)
            except CameraCalibrationError as error:
              reason = str(error)
        return {
            "eligible": reason is None,
            "reason": reason,
            "confidence_score": confidence_score,
            "confident": (
                reason is None
                and confidence_score >= FULL_BODY_CONFIDENCE_THRESHOLD
            ),
            "quality_score": quality_score,
            "relative_height": relative_height,
            "near_camera": near_camera,
            "track_box": track_box,
        }

    @staticmethod
    def _estimate_calibrated_height(
      calibration_data: object,
      pose: PoseDetection,
      image: np.ndarray,
    ) -> HeightEstimate:
      calibration = (
        calibration_data
        if isinstance(calibration_data, CameraCalibration)
        else CameraCalibration.from_dict(calibration_data)
      )
      points = {point.index: point for point in pose.keypoints}
      if 0 not in points or 15 not in points or 16 not in points:
        raise CameraCalibrationError(
          "Calibrated height needs nose and both ankle keypoints."
        )
      box = pose.bounding_box
      head_pixel = ((box.x1 + box.x2) / 2, box.y1)
      foot_pixel = (
        (points[15].x + points[16].x) / 2,
        box.y2,
      )
      frame_height, frame_width = image.shape[:2]
      return calibration.estimate_height_cm(
        head_pixel,
        foot_pixel,
        (frame_width, frame_height),
      )

    def _analyze_payload(self, payload: object) -> dict:
        if not isinstance(payload, dict):
            raise TypeError("Request body must be a JSON object.")
        demo = payload.get("mode") == "video_demo"
        calibration_data = payload.get("camera_calibration")
        calibrated_demo = demo and calibration_data is not None
        simulated_demo = demo and calibration_data is None
        camera_geometry_demo = demo
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
        face_detection = None
        if self.face_detector is None:
          self.__class__.face_detector = FaceHeightDetector()
        face_detector = cast(FaceHeightDetector, self.face_detector)
        face_detection = face_detector.detect(image, front_pose)

        camera_height_estimate = None
        if camera_geometry_demo:
            camera_model = (
                CameraCalibration.from_dict(calibration_data)
                if calibrated_demo
                else CameraCalibration.simulated_for_image(
                    (image.shape[1], image.shape[0])
                )
            )
            camera_height_estimate = self._estimate_calibrated_height(
            camera_model,
                front_pose,
                image,
            )
            frontal_span_cm = None
            shoulder_width = None
            hip_width = None
            face_height_cm = None
            box_height_cm = camera_height_estimate.height_cm
            height_cm = camera_height_estimate.height_cm
        else:
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
        if not camera_geometry_demo and not 120 <= height_cm <= 230:
            raise ValueError(
                "Full-body detection-box height is outside 120–230 cm. Check that "
                "the frontal image shows the entire person and the marker is "
                "correctly calibrated."
            )
        if camera_geometry_demo and not 80 <= height_cm <= 260:
          raise ValueError("Camera-geometry height is outside the supported 80-260 cm range.")
        width_ratio = front_pose.bounding_box.width / max(
            front_pose.bounding_box.height, 1.0
        )
        build_category = (
            "athletic" if width_ratio < 0.32 else "average" if width_ratio < 0.43 else "robust"
        )
        size_class = "small" if height_cm < 160 else "medium" if height_cm <= 180 else "large"
        image_height, image_width = image.shape[:2]
        box_left = max(0.0, min(float(image_width), front_pose.bounding_box.x1))
        box_top = max(0.0, min(float(image_height), front_pose.bounding_box.y1))
        box_right = max(box_left, min(float(image_width), front_pose.bounding_box.x2))
        box_bottom = max(box_top, min(float(image_height), front_pose.bounding_box.y2))
        morphology_figure_keypoints = {
          point.index: {
            "x": round(point.x / image_width * 100, 4),
            "y": round(point.y / image_height * 100, 4),
          }
          for point in front_pose.keypoints
          if point.confidence >= Settings().keypoint_threshold
        }
        morphology_proportions = None
        if frontal is not None:
          upper_leg_ratio = frontal.upper_leg_length_over_span
          lower_leg_ratio = frontal.lower_leg_length_over_span
          morphology_proportions = {
            "view": "frontal",
            "reference_span_px": frontal.face_to_ankle_span_px,
            "reference_span": "face-center to ankle-center image distance",
            "shoulder_width_over_span": frontal.shoulder_width_over_span,
            "hip_width_over_span": frontal.hip_width_over_span,
            "torso_length_over_span": frontal.torso_length_over_span,
            "upper_arm_length_over_span": frontal.upper_arm_length_over_span,
            "lower_arm_length_over_span": frontal.lower_arm_length_over_span,
            "upper_leg_length_over_span": upper_leg_ratio,
            "lower_leg_length_over_span": lower_leg_ratio,
            "total_leg_length_over_span": (
              round(upper_leg_ratio + lower_leg_ratio, 4)
              if upper_leg_ratio is not None and lower_leg_ratio is not None
              else None
            ),
            "keypoint_confidence": frontal.keypoint_confidence,
          }
        height_sensitivity_range_cm = None
        sensitivity_heights_cm = None
        if simulated_demo:
          low_mount_height, high_mount_height = SIMULATED_CAMERA_HEIGHT_RANGE_M
          low_height_cm = height_cm * low_mount_height / SIMULATED_CAMERA_HEIGHT_M
          high_height_cm = height_cm * high_mount_height / SIMULATED_CAMERA_HEIGHT_M
          height_sensitivity_range_cm = (
            round(min(low_height_cm, high_height_cm), 1),
            round(max(low_height_cm, high_height_cm), 1),
          )
          sensitivity_heights_cm = (
            low_height_cm,
            high_height_cm,
          )

        morphology_dimensions_cm = None
        morphology_dimension_sensitivity_cm = None
        morphology_reference_span_cm = None
        if morphology_proportions is not None:
          box_height_px = max(front_pose.bounding_box.height, 1.0)
          span_px = morphology_proportions["reference_span_px"]
          morphology_reference_span_cm = span_px * height_cm / box_height_px
          ratio_fields = {
            "shoulder_width": "shoulder_width_over_span",
            "hip_width": "hip_width_over_span",
            "torso_length": "torso_length_over_span",
            "upper_arm_length": "upper_arm_length_over_span",
            "lower_arm_length": "lower_arm_length_over_span",
            "upper_leg_length": "upper_leg_length_over_span",
            "lower_leg_length": "lower_leg_length_over_span",
            "total_leg_length": "total_leg_length_over_span",
          }
          morphology_dimensions_cm = {
            name: self._round_optional(
              ratio * morphology_reference_span_cm
              if ratio is not None
              else None
            )
            for name, ratio_key in ratio_fields.items()
            for ratio in (morphology_proportions[ratio_key],)
          }
          if sensitivity_heights_cm is not None:
            morphology_dimension_sensitivity_cm = {}
            for name, ratio_key in ratio_fields.items():
              ratio = morphology_proportions[ratio_key]
              if ratio is None:
                morphology_dimension_sensitivity_cm[name] = None
                continue
              values = [
                ratio * span_px * estimate / box_height_px
                for estimate in sensitivity_heights_cm
              ]
              morphology_dimension_sensitivity_cm[name] = [
                round(min(values), 1),
                round(max(values), 1),
              ]
          if camera_geometry_demo:
            shoulder_width = morphology_dimensions_cm["shoulder_width"]
            hip_width = morphology_dimensions_cm["hip_width"]

        if calibrated_demo:
          calibration = CameraCalibration.from_dict(calibration_data)
          reprojection_error = (
            f"{calibration.reprojection_rms_px:.3f} px"
            if calibration.reprojection_rms_px is not None
            else "not reported"
          )
          warnings = [
            "Height uses chessboard-calibrated intrinsics and the supplied camera-to-vehicle pose.",
            f"Assumes a level ground plane at vehicle z={calibration.ground_plane_z_m:.2f} m and an upright vertical body.",
            "Head and ground-contact use the person-box top/bottom with ankle-centered ground position; box localization and mount-pose uncertainty are not included.",
            f"Vertical-line consistency residual: {camera_height_estimate.vertical_alignment_residual_m:.3f} m.",
            f"Chessboard RMS reprojection error: {reprojection_error}; this does not validate camera mount measurements.",
                "Morphology centimeters are image-space proportions scaled by the estimated person-box height; they are not independent 3D segment measurements.",
            "Build class is a rough bounding-box-width proxy and needs validation.",
            "The seat movement preview is illustrative, not a validated comfort or driving recommendation.",
          ]
        elif simulated_demo:
          if camera_model.reported_diagonal_fov_deg is not None:
            intrinsics_warnings = [
            f"ESTIMATED INTRINSICS, NOT FULLY CALIBRATED: {CAMERA_IMAGE_SIZE_PX[0]}x{CAMERA_IMAGE_SIZE_PX[1]} pixels, fx={SIMULATED_FX_PX:.1f} px, fy={SIMULATED_FY_PX:.1f} px, cx={SIMULATED_CX_PX:.1f} px, cy={SIMULATED_CY_PX:.1f} px; derived FOV {SIMULATED_HORIZONTAL_FOV_DEG:.2f} x {SIMULATED_VERTICAL_FOV_DEG:.2f} degrees; lens distortion is assumed zero.",
            f"FOV SPECIFICATION MISMATCH: reported diagonal is {REPORTED_DIAGONAL_FOV_DEG:.2f} degrees, but horizontal and vertical FOV imply {IMPLIED_DIAGONAL_FOV_DEG:.2f} degrees for a rectilinear camera.",
            ]
            if image_height > image_width:
              intrinsics_warnings.append(
                f"IMAGE GEOMETRY ASSUMPTION: {image_width}x{image_height} portrait frame is a 90 degree anticlockwise rotation of the full sensor image followed by resizing, possibly with unequal horizontal and vertical scales. Cropping or other processing would invalidate this mapping."
              )
          else:
            intrinsics_warnings = [
              f"SIMULATION ONLY: frame aspect ratio {image_width}x{image_height} does not match the specified {CAMERA_IMAGE_SIZE_PX[0]}x{CAMERA_IMAGE_SIZE_PX[1]} camera. Rotation or cropping is unknown, so a generic 70 degree horizontal FOV is used instead of the mirror-camera specs.",
            ]
          warnings = intrinsics_warnings + [
            f"REPORTED CAMERA POSITION: X 0.74 m aft, Y -0.85 m, Z 0.87 m in the supplied vehicle axes; camera height above ground is {SIMULATED_CAMERA_HEIGHT_M:.2f} m, giving ground Z=-0.28 m.",
            f"MODEL POSE, NOT PHYSICALLY CALIBRATED: effective pitch {SIMULATED_CAMERA_PITCH_DEG:.1f} degrees (downward) and roll {SIMULATED_CAMERA_ROLL_DEG:.1f} degrees fitted to seven videos of three people; pitch differs from the user-supplied -15 degrees. Yaw +{SIMULATED_CAMERA_YAW_DEG:.0f} degrees from vehicle-forward toward vehicle-left is unverified and cannot be fitted from height alone. Searched pitch {CAMERA_PITCH_SEARCH_RANGE_DEG[0]:.0f} to {CAMERA_PITCH_SEARCH_RANGE_DEG[1]:.0f} degrees and roll {CAMERA_ROLL_SEARCH_RANGE_DEG[0]:.0f} to {CAMERA_ROLL_SEARCH_RANGE_DEG[1]:.0f} degrees.",
            f"ILLUSTRATIVE ONLY: sensitivity to camera height {low_mount_height:.2f}-{high_mount_height:.2f} m is {height_sensitivity_range_cm[0]:.1f}-{height_sensitivity_range_cm[1]:.1f} cm. This is not a measured uncertainty interval and excludes camera orientation, FOV inconsistency, lens distortion, vehicle pitch, and pose error.",
            "The height estimate is not a validated physical measurement. Measure camera orientation and calibrate lens distortion before relying on metric height.",
            "Morphology centimeters are image-space proportions scaled by the simulated person-box height; they are not calibrated physical dimensions.",
            "The seat movement preview is illustrative, not a validated comfort or driving recommendation.",
          ]
        else:
          warnings = [
            "Height uses the frontal full-body detection box scaled by the reference marker; the box must include the top of the head and the soles.",
            "Build class is a rough bounding-box-width proxy and needs validation.",
            "Reference and person must be upright, coplanar, and similarly distant from the camera.",
            "Torso and leg measurements are unavailable because this workflow uses one frontal image.",
            "The seat movement preview is illustrative, not a validated comfort or driving recommendation.",
          ]
        if frontal is None:
            warnings.append(
                "The person bounding box was detected, but required pose landmarks "
                "were incomplete; shoulder and hip measurements are unavailable."
            )
        if face_detection is None and not camera_geometry_demo:
            warnings.append(
                "YuNet did not detect a face; face confidence is unavailable."
            )
        return {
            "estimation_mode": (
                "calibrated_camera_geometry"
                if calibrated_demo
                else "simulated_camera_geometry"
                if simulated_demo
                else "reference_marker"
            ),
            "calibration_mode": (
                "vehicle_camera"
                if calibrated_demo
                else "simulated_vehicle_camera"
                if simulated_demo
                else None
            ),
            "height_cm": round(height_cm, 1),
            "height_sensitivity_range_cm": (
                list(height_sensitivity_range_cm)
                if height_sensitivity_range_cm is not None
                else None
            ),
            "height_geometry_residual_m": (
                round(camera_height_estimate.vertical_alignment_residual_m, 4)
                if camera_height_estimate is not None
                else None
            ),
            "morphology_proportions": morphology_proportions,
            "morphology_figure_keypoints": morphology_figure_keypoints,
            "morphology_reference_span_cm": (
              self._round_optional(morphology_reference_span_cm)
            ),
            "morphology_dimensions_cm": morphology_dimensions_cm,
            "morphology_dimension_sensitivity_cm": morphology_dimension_sensitivity_cm,
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


class _HmiServer(HTTPServer):
  allow_reuse_address = not hasattr(socket, "SO_EXCLUSIVEADDRUSE")

  def server_bind(self) -> None:
    if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
      self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    super().server_bind()


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

    try:
      server = _HmiServer(("127.0.0.1", args.port), _HmiHandler)
    except OSError as error:
      parser.exit(
        1,
        f"Cannot start HMI on port {args.port}: {error}. "
        "Stop the existing HMI or choose another port with --port.\n",
      )
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
