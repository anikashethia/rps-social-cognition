/**
 * Full jsPsych timeline builder for the RPS social cognition task.
 *
 * Fetches the participant's rotation config (avatar order + conditions) from
 * the backend, then builds: welcome → rules → (scanner wait) → 8 blocks × trials → end.
 */

import HtmlKeyboardResponsePlugin from "@jspsych/plugin-html-keyboard-response";
import { getRotation, postTrial, postTrigger, setAnchor, type BlockConfig, type TrialData } from "./api";
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
  dev:        { trials: 5,  iti_min: 0, iti_max: 1000, response_window: 4000, feedback: 2000 },
  behavioral: { trials: 20, iti_min: 0, iti_max: 6000, response_window: 4000, feedback: 2000 },
  scanner:    { trials: 20, iti_min: 0, iti_max: 6000, response_window: 4000, feedback: 2000 },
};

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

function updateHUD(blockNum: number, totalBlocks: number, pts: number) {
  const hud = document.getElementById("hud");
  if (!hud) return;
  hud.style.display = "block";
  const blockEl = document.getElementById("hud-block");
  const ptsEl = document.getElementById("hud-pts");
  if (blockEl) blockEl.textContent = `Block ${blockNum} / ${totalBlocks}`;
  if (ptsEl) ptsEl.textContent = `${pts} pts`;
}

function hideHUD() {
  const hud = document.getElementById("hud");
  if (hud) hud.style.display = "none";
}

// ── HTML generators ──────────────────────────────────────────────────────────

function avatarHTML(avatarId: string, size = 110): string {
  return `<img
    class="avatar-img"
    src="/avatars/${avatarId}.png"
    alt="Player"
    width="${size}"
    height="${size}"
    style="border-radius:50%;object-fit:cover;"
    onerror="this.style.display='none'"
  />`;
}

function welcomeHTML(): string {
  return `
    <div class="screen">
      <div class="tag">RPS · Social Cognition Task</div>
      <div class="body-text">
        Welcome to the next part of the study.<br><br>
        You will now play <strong>Rock-Paper-Scissors</strong> against
        several players.
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

function blockIntroHTML(block: BlockConfig, blockNum: number, totalBlocks: number): string {
  const pct = (((blockNum - 1) / totalBlocks) * 100).toFixed(0);
  return `
    <div class="screen">
      <div class="tag">Block ${blockNum} of ${totalBlocks}</div>
      ${avatarHTML(block.avatar_id, 100)}
      <div class="agent-name">Player 1</div>
      <div class="body-text">You will now play against <strong>Player 1</strong>.</div>
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

export async function buildTimeline(
  sessionId: number,
  mode: Mode,
  configIndex: number,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
): Promise<any[]> {
  const rotation = await getRotation(configIndex);
  const timings = TIMINGS[mode];
  const blocks = rotation.blocks;
  const totalBlocks = blocks.length;

  // ── Mutable session state ──────────────────────────────────────────────

  let sessionAnchorMs = 0;
  let points = 100;
  let trialGlobal = 0;
  let trNumber = 1;
  let currentAgent: Agent | null = null;
  let currentBlockOnsetMs = 0;
  let lastItiMs = 0;
  let lastResult: TrialData | null = null;

  // ── F8 trigger listener ────────────────────────────────────────────────

  function setupTriggerListener() {
    document.addEventListener("keydown", (e: KeyboardEvent) => {
      if (e.key !== "F8") return;
      const tMs = performance.now() - sessionAnchorMs;
      postTrigger(sessionId, { tr_number: trNumber++, t_ms: tMs }).catch(console.error);
    });
  }

  // ── Build timeline ─────────────────────────────────────────────────────

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const timeline: any[] = [];

  timeline.push({
    type: HtmlKeyboardResponsePlugin,
    stimulus: welcomeHTML(),
    choices: "ALL_KEYS",
  });

  timeline.push({
    type: HtmlKeyboardResponsePlugin,
    stimulus: rulesHTML(),
    choices: "ALL_KEYS",
  });

  if (mode === "scanner") {
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
    timeline.push({
      type: HtmlKeyboardResponsePlugin,
      stimulus: "",
      choices: "NO_KEYS",
      trial_duration: 1,
      on_start: () => {
        sessionAnchorMs = performance.now();
        if (mode === "dev") setupTriggerListener();
      },
    });
  }

  // ── Blocks ─────────────────────────────────────────────────────────────

  for (let b = 0; b < totalBlocks; b++) {
    const block = blocks[b]!;
    const blockNum = b + 1;

    timeline.push({
      type: HtmlKeyboardResponsePlugin,
      stimulus: () => blockIntroHTML(block, blockNum, totalBlocks),
      choices: "ALL_KEYS",
      on_start: () => {
        currentAgent = buildAgent(block.avatar_id, configIndex * 1000 + b);
        updateHUD(blockNum, totalBlocks, points);
      },
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      on_finish: (_data: any) => {
        currentBlockOnsetMs = performance.now() - sessionAnchorMs;
      },
    });

    for (let t = 0; t < timings.trials; t++) {
      const itiMs = Math.floor(
        Math.random() * (timings.iti_max - timings.iti_min) + timings.iti_min,
      );

      timeline.push({
        type: FixationPlugin,
        duration_ms: itiMs,
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        on_finish: (data: any) => {
          lastItiMs = data.iti_duration_ms as number;
        },
      });

      timeline.push({
        type: RpsChoicePlugin,
        agent_id: block.avatar_id,
        agent_name: "Player 1",
        response_window_ms: timings.response_window,
        onset_ms: () => performance.now() - sessionAnchorMs,
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        on_finish: (data: any) => {
          const pChoice: number | null = data.participant_choice ?? null;
          const aChoice = currentAgent!.choose();

          if (pChoice !== null) currentAgent!.update(aChoice, pChoice);

          const out = pChoice !== null ? computeOutcome(pChoice, aChoice) : "timeout";
          const delta = pChoice !== null ? pointsDelta(out) : 0;
          points += delta;
          trialGlobal++;

          lastResult = {
            block: blockNum,
            agent: block.avatar_id,
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
            condition: block.condition,
          };

          updateHUD(blockNum, totalBlocks, points);
        },
      });

      timeline.push({
        type: FeedbackPlugin,
        participant_choice: () => lastResult?.participant_choice ?? null,
        agent_choice: () => lastResult?.agent_choice ?? 1,
        outcome: () => lastResult?.outcome ?? "timeout",
        points_delta: () => lastResult?.points_delta ?? 0,
        points_cumulative: () => lastResult?.points_cumulative ?? points,
        feedback_duration_ms: timings.feedback,
        on_finish: () => {
          if (lastResult) postTrial(sessionId, lastResult).catch(console.error);
        },
      });
    }
  }

  timeline.push({
    type: HtmlKeyboardResponsePlugin,
    stimulus: () => endHTML(points),
    choices: "ALL_KEYS",
    on_start: () => { hideHUD(); },
  });

  return timeline;
}
