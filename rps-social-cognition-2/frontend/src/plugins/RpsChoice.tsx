/**
 * jsPsych 8 plugin — RPS choice screen.
 *
 * Shows agent avatar + three choice buttons (Rock / Paper / Scissors).
 * Accepts keys 1, 2, 3 only.
 *
 * Critical fMRI requirement: the screen stays visible for the full
 * `response_window_ms` even after a keypress. The keypress and RT are
 * captured immediately but the trial does NOT advance until the window
 * expires.
 *
 * Records: participant_choice (1/2/3 or null), rt_ms, onset_ms.
 */

import { JsPsych, JsPsychPlugin, ParameterType, TrialType } from "jspsych";

const info = {
  name: "rps-choice",
  version: "1.0.0",
  parameters: {
    agent_id: { type: ParameterType.STRING, default: "" },
    agent_name: { type: ParameterType.STRING, default: "" },
    response_window_ms: { type: ParameterType.INT, default: 4000 },
    onset_ms: { type: ParameterType.FLOAT, default: 0 },
  },
  data: {
    participant_choice: { type: ParameterType.INT },
    rt_ms: { type: ParameterType.FLOAT },
    onset_ms: { type: ParameterType.FLOAT },
  },
} as const;

type Info = typeof info;

const EMOJIS: Record<string, string> = {
  agent_a: "🔵",
  agent_b: "🟢",
  agent_c: "🟠",
  agent_d: "🔴",
  rng: "🎲",
};

class RpsChoicePlugin implements JsPsychPlugin<Info> {
  static info = info;

  constructor(private jsPsych: JsPsych) {}

  trial(display_element: HTMLElement, trial: TrialType<Info>) {
    const trialOnset = performance.now();
    const agentId = trial.agent_id ?? "";
    const agentName = trial.agent_name ?? "";
    const responseWindow = trial.response_window_ms ?? 4000;
    const onsetMs = trial.onset_ms ?? 0;
    const emoji = EMOJIS[agentId] ?? "❓";

    display_element.innerHTML = `
      <div class="screen">
        <div class="avatar">${emoji}</div>
        <div class="agent-name">${agentName}</div>
        <div class="choice-prompt">choose your move</div>
        <div class="choice-row">
          <button class="choice-btn" data-choice="1">
            <span class="ci">🪨</span>
            <span class="cn">Rock</span>
            <span class="ck">1</span>
          </button>
          <button class="choice-btn" data-choice="2">
            <span class="ci">📄</span>
            <span class="cn">Paper</span>
            <span class="ck">2</span>
          </button>
          <button class="choice-btn" data-choice="3">
            <span class="ci">✂️</span>
            <span class="cn">Scissors</span>
            <span class="ck">3</span>
          </button>
        </div>
      </div>`;

    let responded = false;
    let choice: number | null = null;
    let rt: number | null = null;

    const handleKey = (e: KeyboardEvent) => {
      if (responded) return;
      if (e.key !== "1" && e.key !== "2" && e.key !== "3") return;

      responded = true;
      choice = parseInt(e.key, 10);
      rt = performance.now() - trialOnset;

      // Visual highlight — light up the selected button
      const btn = display_element.querySelector(
        `[data-choice="${e.key}"]`,
      ) as HTMLElement | null;
      if (btn) {
        btn.style.borderColor = "var(--accent)";
        btn.style.transform = "translateY(-3px)";
        btn.style.boxShadow = "0 4px 20px rgba(108,142,255,0.25)";
      }
    };

    document.addEventListener("keydown", handleKey);

    // End trial ONLY after the full response window — never early.
    this.jsPsych.pluginAPI.setTimeout(() => {
      document.removeEventListener("keydown", handleKey);
      display_element.innerHTML = "";

      this.jsPsych.finishTrial({
        participant_choice: choice,
        rt_ms: rt,
        onset_ms: onsetMs,
        onset_raw: trialOnset,
      });
    }, responseWindow);
  }
}

export default RpsChoicePlugin;
