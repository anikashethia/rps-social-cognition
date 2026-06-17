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
    <div className="flex h-screen items-center justify-center">
      <div className="w-full max-w-md space-y-8 rounded-2xl border border-neutral-700 bg-neutral-800 p-8 shadow-xl">
        {/* Title */}
        <div className="text-center">
          <h1 className="text-2xl font-bold tracking-tight">
            RPS Social Cognition
          </h1>
          <p className="mt-1 text-sm text-neutral-400">
            Rock&ndash;Paper&ndash;Scissors Task
          </p>
        </div>

        {/* Participant ID */}
        <div className="space-y-2">
          <label className="block text-sm text-neutral-300">
            Participant ID
          </label>
          <input
            type="text"
            value={participantId}
            onChange={(e) => setParticipantId(e.target.value)}
            placeholder="e.g. SUB001"
            className="w-full rounded-lg border border-neutral-600 bg-neutral-900 px-4 py-2 text-neutral-100 placeholder-neutral-500 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
          />
        </div>

        {/* Mode selector */}
        <div className="space-y-2">
          <label className="block text-sm text-neutral-300">Mode</label>
          <div className="flex gap-3">
            {MODES.map((m) => (
              <button
                key={m.value}
                onClick={() => setMode(m.value)}
                className={`flex-1 rounded-lg border px-3 py-2 text-sm transition-colors ${
                  mode === m.value
                    ? "border-blue-500 bg-blue-500/20 text-blue-300"
                    : "border-neutral-600 bg-neutral-900 text-neutral-400 hover:border-neutral-500"
                }`}
              >
                {m.label}
              </button>
            ))}
          </div>
        </div>

        {/* Config index */}
        <div className="space-y-2">
          <label className="block text-sm text-neutral-300">
            Config Index{" "}
            <span className="text-neutral-500">(1&ndash;{configMax})</span>
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
            className="w-full rounded-lg border border-neutral-600 bg-neutral-900 px-4 py-2 text-neutral-100 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
          />
        </div>

        {/* Error */}
        {error && (
          <p className="rounded-lg bg-red-500/10 px-4 py-2 text-sm text-red-400">
            {error}
          </p>
        )}

        {/* Begin button */}
        <button
          onClick={handleBegin}
          disabled={status === "launching"}
          className="w-full rounded-lg bg-blue-600 py-3 text-sm font-semibold text-white transition-colors hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {status === "launching" ? "Launching..." : "Begin"}
        </button>
      </div>
    </div>
  );
}
