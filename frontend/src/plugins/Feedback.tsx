/**
 * jsPsych 8 plugin — trial feedback screen.
 *
 * Shows both choices as styled pills, WIN/LOSE/DRAW badge,
 * points delta and cumulative total.
 * Auto-advances after `feedback_duration_ms`; NO_KEYS.
 */

import { JsPsych, JsPsychPlugin, ParameterType, TrialType } from "jspsych";

const info = {
  name: "rps-feedback",
  version: "1.0.0",
  parameters: {
    participant_choice: { type: ParameterType.INT, default: undefined as unknown as number },
    agent_choice: { type: ParameterType.INT, default: 1 },
    outcome: { type: ParameterType.STRING, default: "draw" },
    points_delta: { type: ParameterType.INT, default: 0 },
    points_cumulative: { type: ParameterType.INT, default: 100 },
    feedback_duration_ms: { type: ParameterType.INT, default: 2000 },
  },
  data: {},
} as const;

type Info = typeof info;

const LABELS: Record<number, string> = { 1: "Rock", 2: "Paper", 3: "Scissors" };
const ICONS: Record<number, string> = { 1: "🪨", 2: "📄", 3: "✂️" };
const CLASSES: Record<number, string> = { 1: "rock", 2: "paper", 3: "scissors" };
const OUTCOME_LABELS: Record<string, string> = {
  win: "Win",
  lose: "Lose",
  draw: "Draw",
  timeout: "Time out",
};

class FeedbackPlugin implements JsPsychPlugin<Info> {
  static info = info;

  constructor(private jsPsych: JsPsych) {}

  trial(display_element: HTMLElement, trial: TrialType<Info>) {
    const pc = (trial.participant_choice ?? null) as number | null;
    const ac = (trial.agent_choice ?? 1) as number;
    const outcome = (trial.outcome ?? "timeout") as string;
    const ptsDelta = (trial.points_delta ?? 0) as number;
    const ptsCumulative = (trial.points_cumulative ?? 100) as number;
    const duration = (trial.feedback_duration_ms ?? 2000) as number;

    if (pc === null || pc === 0) {
      // Timeout screen
      display_element.innerHTML = `
        <div class="screen">
          <div style="color:var(--dim);font-size:16px;">
            Time out
          </div>
          <div class="total-line">${ptsCumulative} pts total</div>
        </div>`;
    } else {
      const dc = ptsDelta > 0 ? "pos" : ptsDelta < 0 ? "neg" : "zero";
      const ds = ptsDelta > 0 ? `+${ptsDelta}` : `${ptsDelta}`;

      display_element.innerHTML = `
        <div class="screen">
          <div class="vs-row">
            <div class="vs-side">
              <div class="vs-who">you</div>
              <div class="pill ${CLASSES[pc] ?? ""}">${ICONS[pc] ?? "?"} ${LABELS[pc] ?? "?"}</div>
            </div>
            <div class="vs-sep">vs</div>
            <div class="vs-side">
              <div class="vs-who">agent</div>
              <div class="pill ${CLASSES[ac] ?? ""}">${ICONS[ac] ?? "?"} ${LABELS[ac] ?? "?"}</div>
            </div>
          </div>
          <div class="outcome-badge ${outcome}">${OUTCOME_LABELS[outcome] ?? outcome}</div>
          <div class="delta ${dc}">${ds} pts</div>
          <div class="total-line">${ptsCumulative} pts total</div>
        </div>`;
    }

    this.jsPsych.pluginAPI.setTimeout(() => {
      display_element.innerHTML = "";
      this.jsPsych.finishTrial({});
    }, duration);
  }
}

export default FeedbackPlugin;
