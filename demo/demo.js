// Interactive demo for motornet.js.
import {
  RigidTendonArm26, ReluPointMass24, RandomTargetReach, RigidTendonHillMuscle, PolicyGRU,
} from '../src/index.js';

const canvas = document.getElementById('view');
const ctx = canvas.getContext('2d');
const W = canvas.width;
const H = canvas.height;

const effSel = document.getElementById('effector');
const musclesDiv = document.getElementById('muscles');
const stateDiv = document.getElementById('state');
const policyHelp = document.getElementById('policyHelp');
const manualHint = document.getElementById('manualHint');

let mode = effSel.value;
let effector = null; // raw effector (manual modes)
let env = null; // environment (policy mode)
let policy = null;
let hidden = null;
let obs = null;
let target = null; // [x, y] world coords
let sliders = []; // manual activations
let camera = null;
let holdFrames = 0;

// ---- camera: world coords -> screen pixels --------------------------------
function makeCamera(xmin, xmax, ymin, ymax) {
  const margin = 40;
  const sx = (W - 2 * margin) / (xmax - xmin);
  const sy = (H - 2 * margin) / (ymax - ymin);
  const s = Math.min(sx, sy);
  const ox = margin + (W - 2 * margin - s * (xmax - xmin)) / 2;
  const oy = margin + (H - 2 * margin - s * (ymax - ymin)) / 2;
  return {
    toScreen: (x, y) => [ox + (x - xmin) * s, H - (oy + (y - ymin) * s)],
    toWorld: (px, py) => [xmin + (px - ox) / s, ymin + (H - py - oy) / s],
  };
}

// ---- build the selected model ---------------------------------------------
async function build() {
  mode = effSel.value;
  effector = null; env = null; policy = null; target = null;
  const isPolicy = mode === 'pointmass_policy';
  policyHelp.style.display = isPolicy ? 'block' : 'none';
  manualHint.style.display = isPolicy ? 'none' : 'block';

  if (mode === 'arm26_manual') {
    effector = new RigidTendonArm26({ muscle: new RigidTendonHillMuscle() });
    effector.reset({ jointState: [Math.PI / 4, Math.PI / 2] });
    camera = makeCamera(-0.65, 0.65, -0.1, 0.75);
    sliders = new Array(effector.nMuscles).fill(0);
    buildMuscleUI(effector.muscleName, true);
  } else if (mode === 'pointmass_manual') {
    effector = new ReluPointMass24();
    effector.reset({ jointState: [0, 0] });
    camera = makeCamera(-1.1, 1.1, -1.1, 1.1);
    sliders = new Array(effector.nMuscles).fill(0);
    buildMuscleUI(effector.muscleName, true);
  } else {
    // trained reaching policy on a point mass
    const meta = await fetch('./policy_pointmass.json').then((r) => r.json());
    effector = new ReluPointMass24();
    env = new RandomTargetReach({ effector, maxEpDuration: 1e9 });
    env.reset({ jointState: [0, 0] });
    policy = new PolicyGRU(meta.input_dim, meta.hidden_dim, meta.output_dim).loadWeights(meta.weights);
    hidden = policy.initHidden();
    camera = makeCamera(-1.1, 1.1, -1.1, 1.1);
    newRandomTarget();
    obs = env.getObs(null, true);
    buildMuscleUI(effector.muscleName, false);
  }
}

function buildMuscleUI(names, manual) {
  musclesDiv.innerHTML = '';
  names.forEach((name, i) => {
    const row = document.createElement('div');
    row.className = 'row';
    const label = document.createElement('span');
    label.className = 'name';
    label.textContent = name;
    row.appendChild(label);
    if (manual) {
      const input = document.createElement('input');
      input.type = 'range'; input.min = 0; input.max = 1; input.step = 0.01; input.value = 0;
      input.addEventListener('input', () => { sliders[i] = parseFloat(input.value); });
      row.appendChild(input);
    } else {
      const bar = document.createElement('div');
      bar.className = 'bar';
      const fill = document.createElement('div');
      fill.id = `bar${i}`;
      bar.appendChild(fill);
      row.appendChild(bar);
    }
    musclesDiv.appendChild(row);
  });
}

function newRandomTarget() {
  if (!env) return;
  const j = effector.drawRandomUniformStates(1);
  target = effector.joint2cartesian(j).slice(0, 2);
  env.goal = target.slice();
}

// ---- simulation step ------------------------------------------------------
function update() {
  if (mode === 'pointmass_policy') {
    const out = policy.forward(obs, hidden);
    hidden = out.h;
    const res = env.step(out.u, { deterministic: true });
    obs = res.obs;
    // update activation bars
    const act = effector.states.muscle[0];
    act.forEach((a, i) => {
      const fill = document.getElementById(`bar${i}`);
      if (fill) fill.style.width = `${Math.min(1, a) * 100}%`;
    });
    // when close to target, hold then re-randomize
    const ft = effector.states.fingertip;
    const dist = Math.hypot(ft[0] - target[0], ft[1] - target[1]);
    if (dist < 0.05) { holdFrames++; if (holdFrames > 40) { newRandomTarget(); holdFrames = 0; } }
    else holdFrames = 0;
  } else {
    effector.step(sliders);
    // update force bars (manual mode shows force magnitude)
    const fIdx = effector.forceIndex;
    const forces = effector.states.muscle[fIdx];
    const maxF = Math.max(1, ...effector.muscle.maxIsoForce);
    forces.forEach((f, i) => {
      const fill = document.getElementById(`bar${i}`);
      if (fill) fill.style.width = `${Math.min(1, f / maxF) * 100}%`;
    });
  }
}

// ---- rendering ------------------------------------------------------------
function drawArm() {
  const sk = effector.skeleton;
  const [sho, elb] = effector.states.joint;
  const ex = sk.L1 * Math.cos(sho);
  const ey = sk.L1 * Math.sin(sho);
  const hx = ex + sk.L2 * Math.cos(sho + elb);
  const hy = ey + sk.L2 * Math.sin(sho + elb);
  const p0 = camera.toScreen(0, 0);
  const p1 = camera.toScreen(ex, ey);
  const p2 = camera.toScreen(hx, hy);

  ctx.lineCap = 'round';
  ctx.strokeStyle = '#30363d'; ctx.lineWidth = 14;
  ctx.beginPath(); ctx.moveTo(...p0); ctx.lineTo(...p1); ctx.lineTo(...p2); ctx.stroke();
  ctx.strokeStyle = '#58a6ff'; ctx.lineWidth = 8;
  ctx.beginPath(); ctx.moveTo(...p0); ctx.lineTo(...p1); ctx.lineTo(...p2); ctx.stroke();
  for (const [p, c] of [[p0, '#8b949e'], [p1, '#c9d1d9'], [p2, '#3fb950']]) {
    ctx.fillStyle = c; ctx.beginPath(); ctx.arc(p[0], p[1], 7, 0, 2 * Math.PI); ctx.fill();
  }
}

function drawPointMass() {
  const ft = effector.states.fingertip;
  const p = camera.toScreen(ft[0], ft[1]);
  // workspace box [-1,1]
  const a = camera.toScreen(-1, -1); const b = camera.toScreen(1, 1);
  ctx.strokeStyle = '#21262d'; ctx.lineWidth = 1;
  ctx.strokeRect(a[0], b[1], b[0] - a[0], a[1] - b[1]);
  ctx.fillStyle = '#58a6ff';
  ctx.beginPath(); ctx.arc(p[0], p[1], 12, 0, 2 * Math.PI); ctx.fill();
}

function drawTarget() {
  if (!target) return;
  const p = camera.toScreen(target[0], target[1]);
  ctx.strokeStyle = '#f0883e'; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(p[0], p[1], 10, 0, 2 * Math.PI); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(p[0] - 14, p[1]); ctx.lineTo(p[0] + 14, p[1]);
  ctx.moveTo(p[0], p[1] - 14); ctx.lineTo(p[0], p[1] + 14); ctx.stroke();
}

function render() {
  ctx.clearRect(0, 0, W, H);
  drawTarget();
  if (effector.skeleton.name === 'two_dof_arm') drawArm();
  else drawPointMass();

  const j = effector.states.joint;
  const ft = effector.states.fingertip;
  stateDiv.innerHTML =
    `joint pos: [${j.slice(0, effector.dof).map((v) => v.toFixed(3)).join(', ')}]<br>` +
    `fingertip: [${ft.map((v) => v.toFixed(3)).join(', ')}]`;
}

// ---- main loop ------------------------------------------------------------
let acc = 0;
let last = performance.now();
function frame(now) {
  const dtReal = (now - last) / 1000; last = now;
  acc += Math.min(dtReal, 0.1);
  const step = effector.dt;
  let n = 0;
  while (acc >= step && n < 20) { update(); acc -= step; n++; }
  render();
  requestAnimationFrame(frame);
}

// ---- interaction ----------------------------------------------------------
canvas.addEventListener('pointerdown', (e) => {
  const rect = canvas.getBoundingClientRect();
  const px = (e.clientX - rect.left) * (W / rect.width);
  const py = (e.clientY - rect.top) * (H / rect.height);
  const [x, y] = camera.toWorld(px, py);
  if (mode === 'pointmass_policy') { target = [x, y]; env.goal = target.slice(); holdFrames = 0; }
});

document.getElementById('resetBtn').addEventListener('click', () => build());
document.getElementById('randomBtn').addEventListener('click', () => {
  if (mode === 'pointmass_policy') newRandomTarget();
});
effSel.addEventListener('change', () => build());

await build();
requestAnimationFrame(frame);
