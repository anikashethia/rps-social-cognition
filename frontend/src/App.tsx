import { useState, useRef, useEffect } from "react";
import { initJsPsych } from "jspsych";
import { createSession, getRegistration, upsertRegistration, type RegistrationOut } from "./api";
import { buildTimeline } from "./timeline";

type Version = "behavioral" | "scanner";
type TestMode = "test" | "full";

const TRIALS_PER_BLOCK: Record<TestMode, number> = { test: 5, full: 40 };

export default function App() {
  const [participantId, setParticipantId] = useState("");
  const [testMode, setTestMode] = useState<TestMode>("test");
  const [configIndex, setConfigIndex] = useState(1);
  const [status, setStatus] = useState<"idle" | "launching" | "running" | "done">("idle");
  const [error, setError] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<number | null>(null);
  const [launch, setLaunch] = useState<{ version: Version } | null>(null);

  // Participant registration state
  const [registration, setRegistration] = useState<RegistrationOut | null>(null);
  const [registrationChecked, setRegistrationChecked] = useState(false);
  const [friendlyId, setFriendlyId] = useState("");
  const [neutralId, setNeutralId] = useState("");
  const [regSaving, setRegSaving] = useState(false);
  const [regError, setRegError] = useState<string | null>(null);

  const configMax = 6;

  const jspsychRef = useRef<HTMLDivElement>(null);
  const initialized = useRef(false);

  // ── Check registration when participant ID is entered ────────────────

  async function checkRegistration(pid: string) {
    if (!pid.trim()) {
      setRegistration(null);
      setRegistrationChecked(false);
      setFriendlyId("");
      setNeutralId("");
      return;
    }
    try {
      const reg = await getRegistration(pid.trim());
      setRegistration(reg);
      setFriendlyId(reg.friendly_avatar_id);
      setNeutralId(reg.neutral_avatar_id);
    } catch {
      setRegistration(null);
      setFriendlyId("");
      setNeutralId("");
    }
    setRegistrationChecked(true);
  }

  async function handleRegister() {
    if (!participantId.trim() || !friendlyId.trim() || !neutralId.trim()) {
      setRegError("All three fields are required");
      return;
    }
    setRegSaving(true);
    setRegError(null);
    try {
      const reg = await upsertRegistration(participantId.trim(), friendlyId.trim(), neutralId.trim());
      setRegistration(reg);
    } catch (err) {
      setRegError(err instanceof Error ? err.message : "Failed to save registration");
    } finally {
      setRegSaving(false);
    }
  }

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

    buildTimeline(sessionId, launch.version, TRIALS_PER_BLOCK[testMode])
      .then((timeline) => jsPsych.run(timeline))
      .catch((err) => console.error("Failed to build timeline:", err));
  }, [status, sessionId, launch, testMode]);

  // ── Handlers ─────────────────────────────────────────────────────────

  async function handleStart(version: Version) {
    if (!participantId.trim()) {
      setError("Participant ID is required");
      return;
    }
    if (!registration) {
      setError("Register the participant's avatars before starting");
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
      setLaunch({ version });
      setStatus("running");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create session");
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
            Rock&ndash;Paper&ndash;Scissors Task &middot; 6 blocks &middot; 40 trials/block
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
            onChange={(e) => {
              setParticipantId(e.target.value);
              setRegistrationChecked(false);
              setRegistration(null);
            }}
            onBlur={(e) => checkRegistration(e.target.value)}
            placeholder="e.g. SUB001"
            className="w-full rounded-lg border border-slate-200 bg-white px-4 py-2 text-slate-900 placeholder-slate-400 focus:border-slate-400 focus:outline-none focus:ring-1 focus:ring-slate-400"
          />
        </div>

        {/* Avatar Registration */}
        {participantId.trim() && (
          <div className="rounded-lg border border-slate-200 p-4 space-y-3">
            <div className="text-xs font-semibold uppercase tracking-widest text-slate-500">
              Avatar Assignment (from IOS ratings)
            </div>

            {registration ? (
              <div className="space-y-2">
                <div className="rounded bg-green-50 px-3 py-2 text-sm text-green-800">
                  Registered &mdash; Friendly: <strong>{registration.friendly_avatar_id}</strong>
                  &nbsp;&middot;&nbsp; Neutral: <strong>{registration.neutral_avatar_id}</strong>
                </div>
                <button
                  onClick={() => setRegistration(null)}
                  className="text-xs text-slate-400 underline hover:text-slate-600"
                >
                  Edit assignment
                </button>
              </div>
            ) : (
              <div className="space-y-2">
                {registrationChecked && (
                  <p className="text-xs text-amber-600">
                    No registration found &mdash; enter avatar IDs from the chat task IOS data.
                  </p>
                )}
                <div className="flex gap-2">
                  <div className="flex-1 space-y-1">
                    <label className="text-xs text-slate-500">Friendly avatar (highest IOS)</label>
                    <input
                      type="text"
                      value={friendlyId}
                      onChange={(e) => setFriendlyId(e.target.value)}
                      placeholder="e.g. s1f1"
                      className="w-full rounded-lg border border-slate-200 px-3 py-1.5 text-sm text-slate-900 focus:border-slate-400 focus:outline-none focus:ring-1 focus:ring-slate-400"
                    />
                  </div>
                  <div className="flex-1 space-y-1">
                    <label className="text-xs text-slate-500">Neutral avatar (lowest IOS)</label>
                    <input
                      type="text"
                      value={neutralId}
                      onChange={(e) => setNeutralId(e.target.value)}
                      placeholder="e.g. s1m3"
                      className="w-full rounded-lg border border-slate-200 px-3 py-1.5 text-sm text-slate-900 focus:border-slate-400 focus:outline-none focus:ring-1 focus:ring-slate-400"
                    />
                  </div>
                </div>
                {regError && (
                  <p className="text-xs text-red-600">{regError}</p>
                )}
                <button
                  onClick={handleRegister}
                  disabled={regSaving}
                  className="w-full rounded-lg bg-slate-700 py-1.5 text-sm font-semibold text-white hover:bg-slate-600 disabled:opacity-50"
                >
                  {regSaving ? "Saving..." : "Register participant"}
                </button>
              </div>
            )}
          </div>
        )}

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
              Test (5 trials/block)
            </label>
            <label className="flex items-center gap-2 text-sm text-slate-900">
              <input
                type="radio"
                name="test-mode"
                checked={testMode === "full"}
                onChange={() => setTestMode("full")}
                className="h-4 w-4 accent-blue-600"
              />
              Full (40 trials/block)
            </label>
          </div>
        </div>

        {/* Config index (block order counterbalancing) */}
        <div className="space-y-1.5">
          <label className="block text-xs font-semibold uppercase tracking-widest text-slate-500">
            Block Order Config (1–{configMax})
          </label>
          <input
            type="number"
            min={1}
            max={configMax}
            value={configIndex}
            onChange={(e) =>
              setConfigIndex(Math.max(1, Math.min(configMax, Number(e.target.value))))
            }
            className="w-24 rounded-lg border border-slate-200 bg-white px-4 py-2 text-slate-900 focus:border-slate-400 focus:outline-none focus:ring-1 focus:ring-slate-400"
          />
        </div>

        {/* Error */}
        {error && (
          <p className="rounded-lg bg-red-50 px-4 py-2 text-sm text-red-600">
            {error}
          </p>
        )}

        {/* Start buttons */}
        <div className="space-y-2">
          <button
            onClick={() => handleStart("behavioral")}
            disabled={status === "launching" || !registration}
            className="w-full rounded-lg bg-slate-900 py-2 text-sm font-semibold text-white transition-colors hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {status === "launching" ? "Launching..." : "Start online version"}
          </button>
          <button
            onClick={() => handleStart("scanner")}
            disabled={status === "launching" || !registration}
            className="w-full rounded-lg bg-indigo-700 py-2 text-sm font-semibold text-white transition-colors hover:bg-indigo-600 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {status === "launching" ? "Launching..." : "Start scanner version"}
          </button>
        </div>
      </div>
    </div>
  );
}
