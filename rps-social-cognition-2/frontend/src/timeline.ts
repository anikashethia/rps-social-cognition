/**
 * Full jsPsych timeline builder for the RPS social cognition task.
 *
 * Builds welcome → rules → (scanner wait) → blocks × trials → end.
 * Each trial posts data to the backend via api.ts.
 */

import HtmlKeyboardResponsePlugin from "@jspsych/plugin-html-keyboard-response";
import { postTrial, postTrigger, setAnchor, type TrialData } from "./api";
import { buildAgent, type Agent } from "./agents";
import FixationPlugin from "./plugins/Fixation";
import RpsChoicePlugin from "./plugins/RpsChoice";
import FeedbackPlugin from "./plugins/Feedback";

// ── Mode timings ─────────────────────────────────────────────────────────────

type Mode = "dev" | "behavioral" | "scanner";

const TIMINGS: Record<Mode, {
  trials: number;
  iti_min: number;
  iti_max: number;
  response_window: number;
  feedback: number;
}> = {
  dev:        { trials: 5,  iti_min: 0, iti_max: 1000,  response_window: 4000, feedback: 2000 },
  behavioral: { trials: 40, iti_min: 0, iti_max: 6000,  response_window: 4000, feedback: 2000 },
  scanner:    { trials: 40, iti_min: 0, iti_max: 6000,  response_window: 4000, feedback: 2000 },
};

// ── Agent display info ───────────────────────────────────────────────────────

const AGENT_NAMES: Record<string, string> = {
  agent_a: "Agent A",
  agent_b: "Agent B",
  agent_c: "Agent C",
  agent_d: "Agent D",
  rng: "Random Draw",
};

const AGENT_EMOJIS: Record<string, string> = {
  agent_a: "🔵",
  agent_b: "🟢",
  agent_c: "🟠",
  agent_d: "🔴",
  rng: "🎲",
};

// ── Counterbalancing (Latin-square rotations) ────────────────────────────────
// 24 configs: all permutations of 4 social agents, RNG inserted at rotating pos.

function generateRotations(): string[][] {
  const social = ["agent_a", "agent_b", "agent_c", "agent_d"];
  const perms: string[][] = [];

  function permute(arr: string[], l: number) {
    if (l === arr.length - 1) {
      perms.push([...arr]);
      return;
    }
    for (let i = l; i < arr.length; i++) {
      [arr[l]!, arr[i]!] = [arr[i]!, arr[l]!];
      permute(arr, l + 1);
      [arr[l]!, arr[i]!] = [arr[i]!, arr[l]!];
    }
  }

  permute(social, 0);

  return perms.map((perm, i) => {
    const pos = i % 5;
    return [...perm.slice(0, pos), "rng", ...perm.slice(pos)];
  });
}

const ROTATIONS = generateRotations(); // length = 24

// ── Game logic ───────────────────────────────────────────────────────────────

function computeOutcome(pChoice: number, aChoice: number): string {
  if (pChoice === aChoice) return "draw";
  if (pChoice === (aChoice % 3) + 1) return "win";
  return "lose";
}

function pointsDelta(outcome: string): number {
  if (outcome === "win") return 3;
  if (outcome === "lose") return -3;
  return 0;
}

// ── HUD ──────────────────────────────────────────────────────────────────────

function updateHUD(agentId: string, blockNum: number, totalBlocks: number, pts: number) {
  const hud = document.getElementById("hud");
  if (!hud) return;
  hud.style.display = "block";
  const agentEl = document.getElementById("hud-agent");
  const blockEl = document.getElementById("hud-block");
  const ptsEl = document.getElementById("hud-pts");
  if (agentEl) agentEl.textContent = AGENT_NAMES[agentId] ?? "";
  if (blockEl) blockEl.textContent = `Block ${blockNum} / ${totalBlocks}`;
  if (ptsEl) ptsEl.textContent = `${pts} pts`;
}

function hideHUD() {
  const hud = document.getElementById("hud");
  if (hud) hud.style.display = "none";
}

// ── HTML generators ──────────────────────────────────────────────────────────

function avatarHTML(agentId: string, size = 110): string {
  const emoji = AGENT_EMOJIS[agentId] ?? "❓";
  return `<div class="avatar" style="width:${size}px;height:${size}px">${emoji}</div>`;
}

function welcomeHTML(): string {
  return `
    <div class="screen">
      <div class="tag">RPS · Social Cognition Task</div>
      <div class="body-text">
        Welcome to the next part of the study.<br><br>
        You will now play <strong>Rock-Paper-Scissors</strong> against
        each of the agents you interacted with earlier, plus a random draw.
      </div>
      <div class="hint">press any key to continue</div>
    </div>`;
}

function rulesHTML(): string {
  return `
    <div class="screen">
      <div class="tag">How to play</div>
      <div class="rules-grid">
        <div class="rule-card"><div class="icon">🪨</div>Rock<br>beats<br>Scissors</div>
        <div class="rule-card"><div class="icon">✂️</div>Scissors<br>beats<br>Paper</div>
        <div class="rule-card"><div class="icon">📄</div>Paper<br>beats<br>Rock</div>
      </div>
      <div class="body-text">Use keys <strong>1</strong>, <strong>2</strong>, <strong>3</strong> to choose.</div>
      <div class="pts-row">
        <div class="pts-item"><div class="pts-val pos">+3</div><div class="pts-lbl">Win</div></div>
        <div class="pts-item"><div class="pts-val neg">−3</div><div class="pts-lbl">Lose</div></div>
        <div class="pts-item"><div class="pts-val zero">0</div><div class="pts-lbl">Draw</div></div>
      </div>
      <div class="hint">press any key to begin</div>
    </div>`;
}

function waitingForScannerHTML(): string {
  return `
    <div class="screen">
      <div class="tag">Scanner Sync</div>
      <div class="body-text" style="font-size:20px;">
        Waiting for scanner&hellip;
      </div>
      <div class="hint">task will begin on first TR (F8)</div>
    </div>`;
}

function blockIntroHTML(agentId: string, blockNum: number, totalBlocks: number): string {
  const pct = (((blockNum - 1) / totalBlocks) * 100).toFixed(0);
  const name = AGENT_NAMES[agentId] ?? agentId;
  const desc =
    agentId === "rng"
      ? "In this block there is <strong>no opponent</strong>.<br>A choice will be drawn randomly."
      : `You will now play against <strong>${name}</strong>.`;

  return `
    <div class="screen">
      <div class="tag">Block ${blockNum} of ${totalBlocks}</div>
      ${avatarHTML(agentId, 100)}
      <div class="agent-name">${name}</div>
      <div class="body-text">${desc}</div>
      <div class="progress-wrap">
        <div class="progress-bar"><div class="progress-fill" style="width:${pct}%"></div></div>
        <div class="progress-label">${blockNum - 1} / ${totalBlocks} blocks done</div>
      </div>
      <div class="hint">press any key when ready</div>
    </div>`;
}

function endHTML(pts: number): string {
  return `
    <div class="screen">
      <div class="tag">Task complete</div>
      <div class="final-score">${pts}</div>
      <div class="hint" style="margin-top:-12px;">total points</div>
      <div class="body-text">Thank you for participating!<br>Your data has been saved.</div>
      <div class="hint">press any key to finish</div>
    </div>`;
}

// ── Timeline builder ─────────────────────────────────────────────────────────

export function buildTimeline(
  sessionId: number,
  mode: Mode,
  configIndex: number,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
): any[] {
  const timings = TIMINGS[mode];
  const blockOrder = ROTATIONS[(configIndex - 1) % ROTATIONS.length]!;
  const totalBlocks = blockOrder.length;

  // ── Mutable session state (closed over by timeline callbacks) ──────────

  let sessionAnchorMs = 0;
  let points = 100;
  let trialGlobal = 0;
  let trNumber = 1;
  let currentAgent: Agent | null = null;
  let currentBlockOnsetMs = 0;
  let lastItiMs = 0;
  let lastResult: TrialData | null = null;

  // ── F8 trigger listener setup ──────────────────────────────────────────

  function setupTriggerListener() {
    document.addEventListener("keydown", (e: KeyboardEvent) => {
      if (e.key !== "F8") return;
      const tMs = performance.now() - sessionAnchorMs;
      postTrigger(sessionId, { tr_number: trNumber++, t_ms: tMs }).catch(
        console.error,
      );
    });
  }

  // ── Build timeline array ───────────────────────────────────────────────
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const timeline: any[] = [];

  // Welcome
  timeline.push({
    type: HtmlKeyboardResponsePlugin,
    stimulus: welcomeHTML(),
    choices: "ALL_KEYS",
  });

  // Rules
  timeline.push({
    type: HtmlKeyboardResponsePlugin,
    stimulus: rulesHTML(),
    choices: "ALL_KEYS",
  });

  // Anchor setup
  if (mode === "scanner") {
    // Waiting-for-scanner screen — advances only on F8
    timeline.push({
      type: HtmlKeyboardResponsePlugin,
      stimulus: waitingForScannerHTML(),
      choices: ["F8"],
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      on_finish: (_data: any) => {
        sessionAnchorMs = performance.now();
        setAnchor(sessionId, 0).catch(console.error);
        setupTriggerListener();
      },
    });
  } else {
    // Non-scanner: invisible anchor trial + optional dev F8 listener
    timeline.push({
      type: HtmlKeyboardResponsePlugin,
      stimulus: "",
      choices: "NO_KEYS",
      trial_duration: 1,
      on_start: () => {
        sessionAnchorMs = performance.now();
        if (mode === "dev") {
          setupTriggerListener();
        }
      },
    });
  }

  // ── Blocks ─────────────────────────────────────────────────────────────

  for (let b = 0; b < totalBlocks; b++) {
    const agentId = blockOrder[b]!;
    const blockNum = b + 1;

    // Block intro
    timeline.push({
      type: HtmlKeyboardResponsePlugin,
      stimulus: () => blockIntroHTML(agentId, blockNum, totalBlocks),
      choices: "ALL_KEYS",
      on_start: () => {
        currentAgent = buildAgent(agentId, configIndex * 1000 + b);
        updateHUD(agentId, blockNum, totalBlocks, points);
      },
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      on_finish: (_data: any) => {
        currentBlockOnsetMs = performance.now() - sessionAnchorMs;
      },
    });

    // Trials within block
    for (let t = 0; t < timings.trials; t++) {
      // Pre-draw ITI duration (uniform iti_min to iti_max)
      const itiMs = Math.floor(Math.random() * (timings.iti_max - timings.iti_min) + timings.iti_min);

      // ── Fixation ──
      timeline.push({
        type: FixationPlugin,
        duration_ms: itiMs,
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        on_finish: (data: any) => {
          lastItiMs = data.iti_duration_ms as number;
        },
      });

      // ── Choice ──
      timeline.push({
        type: RpsChoicePlugin,
        agent_id: agentId,
        agent_name: AGENT_NAMES[agentId] ?? agentId,
        response_window_ms: timings.response_window,
        // onset_ms is evaluated as a function at trial start time
        onset_ms: () => performance.now() - sessionAnchorMs,
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        on_finish: (data: any) => {
          const pChoice: number | null = data.participant_choice ?? null;
          const aChoice = currentAgent!.choose();

          if (pChoice !== null) {
            currentAgent!.update(aChoice, pChoice);
          }

          const out = pChoice !== null ? computeOutcome(pChoice, aChoice) : "timeout";
          const delta = pChoice !== null ? pointsDelta(out) : 0;
          points += delta;
          trialGlobal++;

          lastResult = {
            block: blockNum,
            agent: agentId,
            trial_in_block: t + 1,
            trial_global: trialGlobal,
            participant_choice: pChoice,
            agent_choice: aChoice,
            outcome: out,
            points_delta: delta,
            points_cumulative: points,
            rt_ms: (data.rt_ms as number | null) ?? null,
            onset_ms: data.onset_ms as number,
            iti_duration_ms: lastItiMs,
            block_onset_ms: currentBlockOnsetMs,
          };

          // Update HUD points
          updateHUD(agentId, blockNum, totalBlocks, points);
        },
      });

      // ── Feedback ──
      timeline.push({
        type: FeedbackPlugin,
        participant_choice: () => lastResult?.participant_choice ?? null,
        agent_choice: () => lastResult?.agent_choice ?? 1,
        outcome: () => lastResult?.outcome ?? "timeout",
        points_delta: () => lastResult?.points_delta ?? 0,
        points_cumulative: () => lastResult?.points_cumulative ?? points,
        feedback_duration_ms: timings.feedback,
        on_finish: () => {
          // POST trial to backend (fire-and-forget for timing)
          if (lastResult) {
            postTrial(sessionId, lastResult).catch(console.error);
          }
        },
      });
    }
  }

  // ── End screen ─────────────────────────────────────────────────────────

  timeline.push({
    type: HtmlKeyboardResponsePlugin,
    stimulus: () => endHTML(points),
    choices: "ALL_KEYS",
    on_start: () => {
      hideHUD();
    },
  });

  return timeline;
}
