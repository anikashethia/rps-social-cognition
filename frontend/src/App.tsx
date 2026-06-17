import { useState, useRef, useEffect } from "react";
import { initJsPsych } from "jspsych";
import { createSession } from "./api";
import { buildTimeline } from "./timeline";

type Mode = "dev" | "behavioral" | "scanner";

const MODES: { value: Mode; label: string }[] = [
  { value: "dev", label: "Dev" },
  { value: "behavioral", label: "Behavioral" },
  { value: "scanner", label: "Scanner" },
];

export default function App() {
  const [participantId, setParticipantId] = useState("");
  const [configIndex, setConfigIndex] = useState(1);
  const [mode, setMode] = useState<Mode>("dev");
  const [status, setStatus] = useState<
    "idle" | "launching" | "running" | "done"
  >("idle");
  const [error, setError] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<number | null>(null);

  const jspsychRef = useRef<HTMLDivElement>(null);
  const initialized = useRef(false);

  const configMax = 16; // expand when mentor's final rotation table is ready

  // ── Launch jsPsych once status flips to "running" ────────────────────

  useEffect(() => {
    if (status !== "running" || sessionId === null || !jspsychRef.current) return;
    if (initialized.current) return;
    initialized.current = true;

    const jsPsych = initJsPsych({
      display_element: jspsychRef.current,
      on_finish: () => {
        setStatus("done");
      },
    });

    buildTimeline(sessionId, mode, configIndex)
      .then((timeline) => jsPsych.run(timeline))
      .catch((err) => console.error("Failed to build timeline:", err));
  }, [status, sessionId, mode, configIndex]);

  // ── Handlers ─────────────────────────────────────────────────────────

  async function handleBegin() {
    if (!participantId.trim()) {
      setError("Participant ID is required");
      return;
    }
    setError(null);
    setStatus("launching");

    try {
      const session = await createSession({
        participant_id: participantId.trim(),
        mode,
        config_index: configIndex,
      });
      setSessionId(session.session_id);
      setStatus("running");
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to create session",
      );
      setStatus("idle");
    }
  }

  // ── Running state: jsPsych display + HUD ─────────────────────────────

  if (status === "running" || status === "done") {
    return (
      <>
        <div ref={jspsychRef} id="jspsych-target" />
        <div id="hud" style={{ display: "none" }}>
          <div id="hud-agent"></div>
          <div id="hud-block"></div>
          <div id="hud-pts"></div>
        </div>
      </>
    );
  }

  // ── Landing page ─────────────────────────────────────────────────────

  return (
    <div className="flex min-h-screen items-center justify-center bg-slate-100 px-4">
      <div className="w-full max-w-xl space-y-6 rounded-2xl border border-slate-200 bg-white p-10 shadow-sm">
        {/* Title */}
        <div className="text-center">
          <h1 className="text-2xl font-bold tracking-tight text-slate-900">
            RPS Social Cognition
          </h1>
          <p className="mt-1 text-sm text-slate-500">
            Rock&ndash;Paper&ndash;Scissors Task
          </p>
        </div>

        {/* Participant ID */}
        <div className="space-y-1.5">
          <label className="block text-xs font-semibold uppercase tracking-widest text-slate-500">
            Participant ID
          </label>
          <input
            type="text"
            value={participantId}
            onChange={(e) => setParticipantId(e.target.value)}
            placeholder="e.g. SUB001"
            className="w-full rounded-lg border border-slate-200 bg-white px-4 py-2 text-slate-900 placeholder-slate-400 focus:border-slate-400 focus:outline-none focus:ring-1 focus:ring-slate-400"
          />
        </div>

        {/* Mode selector */}
        <div className="space-y-1.5">
          <label className="block text-xs font-semibold uppercase tracking-widest text-slate-500">
            Mode
          </label>
          <div className="flex gap-2">
            {MODES.map((m) => (
              <button
                key={m.value}
                onClick={() => setMode(m.value)}
                className={`flex-1 rounded-lg border px-3 py-2 text-sm transition-colors ${
                  mode === m.value
                    ? "border-slate-900 bg-slate-900 text-white"
                    : "border-slate-200 bg-white text-slate-600 hover:border-slate-400"
                }`}
              >
                {m.label}
              </button>
            ))}
          </div>
        </div>

        {/* Config index */}
        <div className="space-y-1.5">
          <label className="block text-xs font-semibold uppercase tracking-widest text-slate-500">
            Config Index{" "}
            <span className="font-normal normal-case tracking-normal text-slate-400">(1&ndash;{configMax})</span>
          </label>
          <input
            type="number"
            min={1}
            max={configMax}
            value={configIndex}
            onChange={(e) =>
              setConfigIndex(
                Math.max(1, Math.min(configMax, Number(e.target.value))),
              )
            }
            className="w-full rounded-lg border border-slate-200 bg-white px-4 py-2 text-slate-900 focus:border-slate-400 focus:outline-none focus:ring-1 focus:ring-slate-400"
          />
        </div>

        {/* Error */}
        {error && (
          <p className="rounded-lg bg-red-50 px-4 py-2 text-sm text-red-600">
            {error}
          </p>
        )}

        {/* Begin button */}
        <button
          onClick={handleBegin}
          disabled={status === "launching"}
          className="w-full rounded-lg bg-slate-900 py-3 text-sm font-semibold text-white transition-colors hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {status === "launching" ? "Launching..." : "Begin"}
        </button>
      </div>
    </div>
  );
}
