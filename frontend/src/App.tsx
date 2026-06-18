import { useState, useRef, useEffect } from "react";
import { initJsPsych } from "jspsych";
import { createSession } from "./api";
import { buildTimeline } from "./timeline";

type Version = "behavioral" | "scanner";
type TestMode = "test" | "full";

const TRIALS_PER_BLOCK: Record<TestMode, number> = { test: 5, full: 35 };

export default function App() {
  const [participantId, setParticipantId] = useState("");
  const [testMode, setTestMode] = useState<TestMode>("test");
  const [onlineConfigIndex, setOnlineConfigIndex] = useState(1);
  const [scannerConfigIndex, setScannerConfigIndex] = useState(1);
  const [status, setStatus] = useState<
    "idle" | "launching" | "running" | "done"
  >("idle");
  const [error, setError] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<number | null>(null);
  const [launch, setLaunch] = useState<{ version: Version; configIndex: number } | null>(null);

  const jspsychRef = useRef<HTMLDivElement>(null);
  const initialized = useRef(false);

  const configMax = 16; // expand when mentor's final rotation table is ready

  // ── Launch jsPsych once status flips to "running" ────────────────────

  useEffect(() => {
    if (status !== "running" || sessionId === null || !launch || !jspsychRef.current) return;
    if (initialized.current) return;
    initialized.current = true;

    const jsPsych = initJsPsych({
      display_element: jspsychRef.current,
      on_finish: () => {
        setStatus("done");
      },
    });

    buildTimeline(sessionId, launch.version, launch.configIndex, TRIALS_PER_BLOCK[testMode])
      .then((timeline) => jsPsych.run(timeline))
      .catch((err) => console.error("Failed to build timeline:", err));
  }, [status, sessionId, launch, testMode]);

  // ── Handlers ─────────────────────────────────────────────────────────

  async function handleStart(version: Version, configIndex: number) {
    if (!participantId.trim()) {
      setError("Participant ID is required");
      return;
    }
    setError(null);
    setStatus("launching");

    try {
      const session = await createSession({
        participant_id: participantId.trim(),
        mode: version,
        config_index: configIndex,
      });
      setSessionId(session.session_id);
      setLaunch({ version, configIndex });
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

        {/* Mode (Test / Full) */}
        <div className="rounded-lg border border-slate-200 p-4 text-center">
          <div className="mb-2 text-xs font-semibold uppercase tracking-widest text-slate-500">
            Mode
          </div>
          <div className="flex items-center justify-center gap-6">
            <label className="flex items-center gap-2 text-sm text-slate-900">
              <input
                type="radio"
                name="test-mode"
                checked={testMode === "test"}
                onChange={() => setTestMode("test")}
                className="h-4 w-4 accent-blue-600"
              />
              Test
            </label>
            <label className="flex items-center gap-2 text-sm text-slate-900">
              <input
                type="radio"
                name="test-mode"
                checked={testMode === "full"}
                onChange={() => setTestMode("full")}
                className="h-4 w-4 accent-blue-600"
              />
              Full
            </label>
          </div>
        </div>

        {/* Error */}
        {error && (
          <p className="rounded-lg bg-red-50 px-4 py-2 text-sm text-red-600">
            {error}
          </p>
        )}

        {/* Online version */}
        <div className="space-y-1.5">
          <label className="block text-xs font-semibold uppercase tracking-widest text-slate-500">
            Online Version
          </label>
          <div className="flex gap-2">
            <input
              type="number"
              min={1}
              max={configMax}
              value={onlineConfigIndex}
              onChange={(e) =>
                setOnlineConfigIndex(
                  Math.max(1, Math.min(configMax, Number(e.target.value))),
                )
              }
              placeholder={`1–${configMax}`}
              className="w-24 rounded-lg border border-slate-200 bg-white px-4 py-2 text-slate-900 focus:border-slate-400 focus:outline-none focus:ring-1 focus:ring-slate-400"
            />
            <button
              onClick={() => handleStart("behavioral", onlineConfigIndex)}
              disabled={status === "launching"}
              className="flex-1 rounded-lg bg-slate-900 py-2 text-sm font-semibold text-white transition-colors hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {status === "launching" ? "Launching..." : "Start online version"}
            </button>
          </div>
        </div>

        {/* Scanner version */}
        <div className="space-y-1.5">
          <label className="block text-xs font-semibold uppercase tracking-widest text-slate-500">
            Scanner Version
          </label>
          <div className="flex gap-2">
            <input
              type="number"
              min={1}
              max={configMax}
              value={scannerConfigIndex}
              onChange={(e) =>
                setScannerConfigIndex(
                  Math.max(1, Math.min(configMax, Number(e.target.value))),
                )
              }
              placeholder={`1–${configMax}`}
              className="w-24 rounded-lg border border-slate-200 bg-white px-4 py-2 text-slate-900 focus:border-slate-400 focus:outline-none focus:ring-1 focus:ring-slate-400"
            />
            <button
              onClick={() => handleStart("scanner", scannerConfigIndex)}
              disabled={status === "launching"}
              className="flex-1 rounded-lg bg-slate-900 py-2 text-sm font-semibold text-white transition-colors hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {status === "launching" ? "Launching..." : "Start scanner version"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
