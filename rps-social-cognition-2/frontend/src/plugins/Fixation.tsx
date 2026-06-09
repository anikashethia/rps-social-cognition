/**
 * jsPsych 8 plugin — ITI fixation cross.
 * Shows a centred "+" on a dark background for `duration_ms`.
 * Records iti_duration_ms in trial data.
 */

import { JsPsych, JsPsychPlugin, ParameterType, TrialType } from "jspsych";

const info = {
  name: "rps-fixation",
  version: "1.0.0",
  parameters: {
    /** How long (ms) to display the fixation cross. */
    duration_ms: {
      type: ParameterType.INT,
      default: 1000,
    },
  },
  data: {
    /** Echoes the actual duration used. */
    iti_duration_ms: {
      type: ParameterType.FLOAT,
    },
  },
} as const;

type Info = typeof info;

class FixationPlugin implements JsPsychPlugin<Info> {
  static info = info;

  constructor(private jsPsych: JsPsych) {}

  trial(display_element: HTMLElement, trial: TrialType<Info>) {
    const duration = trial.duration_ms ?? 1000;

    display_element.innerHTML =
      '<div class="screen"><div class="fixation">+</div></div>';

    this.jsPsych.pluginAPI.setTimeout(() => {
      display_element.innerHTML = "";
      this.jsPsych.finishTrial({ iti_duration_ms: duration });
    }, duration);
  }
}

export default FixationPlugin;
