# task/rps_task.py
"""
RPS Social Cognition Task — main runner.

Usage:
    python rps_task.py --participant_id SUB001 --session 1

Runs 5 blocks (agents A, B, C, D + RNG), counterbalanced order.
Outputs one CSV per session to data/.

Designed for in-scanner administration (MRI button box, scanner trigger).
Can also be run behaviorally (keyboard).
"""

import argparse
import os
import csv
import time
import random
from datetime import datetime
from typing import List, Dict, Any

from psychopy import visual, core, event, data, logging

from task.config import (
    AGENTS, N_TRIALS_PER_BLOCK, N_PRACTICE_TRIALS,
    OPPONENT_DISPLAY_DURATION, FEEDBACK_DURATION, ITI_MIN, ITI_MAX,
    RESPONSE_TIMEOUT, RESPONSE_KEYS, SCANNER_TRIGGER_KEY,
    POINTS_WIN, POINTS_LOSE, POINTS_DRAW, STARTING_POINTS,
    WINDOW_SIZE, FULLSCREEN, BACKGROUND_COLOR, TEXT_COLOR,
    HIGHLIGHT_COLOR, LOSS_COLOR, DRAW_COLOR,
    FONT, FONT_SIZE_LARGE, FONT_SIZE_MEDIUM, FONT_SIZE_SMALL,
    AVATAR_PATHS, AVATAR_SIZE, DATA_DIR, OUTPUT_COLUMNS,
    get_outcome, ACTION_LABELS, ACTION_KEYS,
)
from task.agents import build_agent
from utils.counterbalancing import get_block_order
from utils.data_io import save_trial


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="RPS Social Cognition Task")
    parser.add_argument("--participant_id", required=True)
    parser.add_argument("--session",        type=int, default=1)
    parser.add_argument("--n_trials",       type=int, default=N_TRIALS_PER_BLOCK)
    parser.add_argument("--practice",       action="store_true")
    parser.add_argument("--scanner",        action="store_true",
                        help="Wait for scanner TTL trigger to start each block")
    parser.add_argument("--fullscreen",     action="store_true", default=FULLSCREEN)
    parser.add_argument("--seed",           type=int, default=None)
    return parser.parse_args()


# ── Window & stimuli setup ────────────────────────────────────────────────────

def build_window(fullscreen: bool) -> visual.Window:
    return visual.Window(
        size       = WINDOW_SIZE,
        fullscr    = fullscreen,
        color      = BACKGROUND_COLOR,
        units      = "pix",
        allowGUI   = False,
    )


def build_stimuli(win: visual.Window) -> Dict[str, Any]:
    """Pre-build all reusable visual stimuli."""
    stim = {}

    stim["fixation"] = visual.TextStim(
        win, text="+", color=TEXT_COLOR, height=FONT_SIZE_LARGE, font=FONT
    )
    stim["choice_prompt"] = visual.TextStim(
        win, text="1 = Rock    2 = Paper    3 = Scissors",
        color=TEXT_COLOR, height=FONT_SIZE_MEDIUM, font=FONT, pos=(0, -200)
    )
    stim["feedback_text"] = visual.TextStim(
        win, text="", color=TEXT_COLOR, height=FONT_SIZE_LARGE, font=FONT
    )
    stim["points_text"] = visual.TextStim(
        win, text="", color=TEXT_COLOR, height=FONT_SIZE_SMALL, font=FONT,
        pos=(0, -300)
    )
    stim["agent_label"] = visual.TextStim(
        win, text="", color=TEXT_COLOR, height=FONT_SIZE_MEDIUM, font=FONT,
        pos=(0, 200)
    )
    stim["instruction_text"] = visual.TextStim(
        win, text="", color=TEXT_COLOR, height=FONT_SIZE_MEDIUM, font=FONT,
        wrapWidth=1400
    )

    # Avatar images — one per agent + RNG icon
    stim["avatars"] = {}
    for agent_id, path in AVATAR_PATHS.items():
        if os.path.exists(path):
            stim["avatars"][agent_id] = visual.ImageStim(
                win, image=path, size=AVATAR_SIZE, pos=(0, 50)
            )
        else:
            # Fallback: labelled circle placeholder
            stim["avatars"][agent_id] = visual.TextStim(
                win, text=f"[{agent_id}]", color=TEXT_COLOR,
                height=FONT_SIZE_LARGE, font=FONT, pos=(0, 50)
            )

    return stim


# ── Instructions ──────────────────────────────────────────────────────────────

INSTRUCTIONS = {
    "welcome": (
        "Welcome to the next part of the study.\n\n"
        "You will now play a game of Rock-Paper-Scissors against each of the "
        "agents you interacted with earlier, as well as a random draw.\n\n"
        "Press any key to continue."
    ),
    "rules": (
        "Rock beats Scissors.\n"
        "Scissors beats Paper.\n"
        "Paper beats Rock.\n\n"
        "Win = +{win} points    Lose = {lose} points    Draw = {draw} points\n\n"
        "Try to win as many points as possible.\n\n"
        "Press any key to begin."
    ).format(win=POINTS_WIN, lose=POINTS_LOSE, draw=POINTS_DRAW),
    "block_start": "You will now play against: {agent_label}\n\nPress any key when ready.",
    "rng_block":   (
        "In this block there is NO opponent.\n"
        "A choice (Rock, Paper, or Scissors) will be drawn randomly.\n\n"
        "Press any key when ready."
    ),
    "scanner_wait": "Waiting for scanner...",
    "end":          "The game is complete.\n\nThank you!\n\nTotal points: {points}",
}

AGENT_DISPLAY_NAMES = {
    "agent_a": "Agent A", "agent_b": "Agent B",
    "agent_c": "Agent C", "agent_d": "Agent D",
    "rng":     "Random Draw",
}


def show_text(win, stim, text: str, wait_key: bool = True):
    stim["instruction_text"].setText(text)
    stim["instruction_text"].draw()
    win.flip()
    if wait_key:
        event.waitKeys()


# ── Trial runner ──────────────────────────────────────────────────────────────

def run_trial(
    win, stim, agent, agent_id: str,
    trial_num: int, block_num: int, block_order: int,
    cumulative_points: int, timer: core.Clock,
) -> Dict[str, Any]:
    """
    Run one RPS trial. Returns a dict of trial data.
    """
    # 1. ITI — fixation cross
    iti_duration = random.uniform(ITI_MIN, ITI_MAX)
    stim["fixation"].draw()
    win.flip()
    core.wait(iti_duration)

    # 2. Opponent display
    if agent_id in stim["avatars"]:
        stim["avatars"][agent_id].draw()
    stim["agent_label"].setText(AGENT_DISPLAY_NAMES[agent_id])
    stim["agent_label"].draw()
    win.flip()
    core.wait(OPPONENT_DISPLAY_DURATION)

    # 3. Choice — self-paced
    if agent_id in stim["avatars"]:
        stim["avatars"][agent_id].draw()
    stim["agent_label"].draw()
    stim["choice_prompt"].draw()
    win.flip()

    timer.reset()
    keys = event.waitKeys(
        maxWait   = RESPONSE_TIMEOUT,
        keyList   = RESPONSE_KEYS + ["escape"],
        timeStamped = timer,
    )

    # Handle escape / timeout
    if keys is None:
        return None  # timeout — caller decides how to handle
    key, rt = keys[0]
    if key == "escape":
        core.quit()

    participant_choice = ACTION_KEYS[key]

    # 4. Agent chooses
    agent_choice = agent.choose(agent.opp_history)
    agent.update(agent_choice, participant_choice)  # agent tracks opponent = participant

    # 5. Compute outcome
    outcome = get_outcome(participant_choice, agent_choice)
    points_delta = {
        "win": POINTS_WIN, "lose": POINTS_LOSE, "draw": POINTS_DRAW
    }[outcome]
    cumulative_points += points_delta

    # 6. Feedback
    feedback_label = {"win": "WIN!", "lose": "LOSE", "draw": "DRAW"}[outcome]
    feedback_color = {"win": HIGHLIGHT_COLOR, "lose": LOSS_COLOR, "draw": DRAW_COLOR}[outcome]
    stim["feedback_text"].setText(
        f"{ACTION_LABELS[participant_choice]}  vs  {ACTION_LABELS[agent_choice]}\n\n"
        f"{feedback_label}   {'+' if points_delta >= 0 else ''}{points_delta} pts"
    )
    stim["feedback_text"].setColor(feedback_color)
    stim["points_text"].setText(f"Total: {cumulative_points} pts")
    stim["feedback_text"].draw()
    stim["points_text"].draw()
    win.flip()
    core.wait(FEEDBACK_DURATION)

    return {
        "participant_id":    None,  # filled by caller
        "session":           None,
        "block":             block_num,
        "block_order":       block_order,
        "agent":             agent_id,
        "trial":             trial_num,
        "trial_global":      None,  # filled by caller
        "participant_choice": participant_choice,
        "agent_choice":      agent_choice,
        "outcome":           outcome,
        "points_delta":      points_delta,
        "points_cumulative": cumulative_points,
        "rt":                round(rt, 4),
        "timestamp":         time.time(),
    }, cumulative_points


# ── Block runner ──────────────────────────────────────────────────────────────

def run_block(
    win, stim, agent_id: str, block_num: int,
    n_trials: int, cumulative_points: int,
    participant_id: str, session: int,
    block_order: int, trial_global_offset: int,
    scanner: bool, seed: Optional[int],
) -> (List[Dict], int):
    """Run one block and return list of trial dicts + updated points."""
    from typing import Optional

    # Block start instruction
    if agent_id == "rng":
        show_text(win, stim, INSTRUCTIONS["rng_block"])
    else:
        show_text(win, stim, INSTRUCTIONS["block_start"].format(
            agent_label=AGENT_DISPLAY_NAMES[agent_id]
        ))

    # Optionally wait for scanner TTL
    if scanner:
        stim["instruction_text"].setText(INSTRUCTIONS["scanner_wait"])
        stim["instruction_text"].draw()
        win.flip()
        event.waitKeys(keyList=[SCANNER_TRIGGER_KEY])

    agent  = build_agent(agent_id, seed=seed)
    timer  = core.Clock()
    trials = []

    for t in range(n_trials):
        result = run_trial(
            win, stim, agent, agent_id,
            trial_num=t + 1, block_num=block_num,
            block_order=block_order,
            cumulative_points=cumulative_points,
            timer=timer,
        )
        if result is None:
            # Timeout — log and continue
            logging.warning(f"Timeout on block {block_num}, trial {t + 1}")
            continue

        trial_data, cumulative_points = result
        trial_data["participant_id"] = participant_id
        trial_data["session"]        = session
        trial_data["trial_global"]   = trial_global_offset + t + 1
        trials.append(trial_data)
        save_trial(trial_data, participant_id, session)

    return trials, cumulative_points


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(DATA_DIR, exist_ok=True)

    win  = build_window(args.fullscreen)
    stim = build_stimuli(win)

    # Welcome & rules
    show_text(win, stim, INSTRUCTIONS["welcome"])
    show_text(win, stim, INSTRUCTIONS["rules"])

    # Practice block (optional)
    if args.practice:
        show_text(win, stim, "You will now do a short practice round.\n\nPress any key to begin.")
        practice_agent = build_agent("rng")
        timer = core.Clock()
        pts   = STARTING_POINTS
        for t in range(N_PRACTICE_TRIALS):
            run_trial(win, stim, practice_agent, "rng", t+1, 0, 0, pts, timer)

    # Counterbalanced block order
    block_order_list = get_block_order(args.participant_id, args.session)

    cumulative_points = STARTING_POINTS
    all_trials        = []
    global_trial      = 0

    for block_idx, agent_id in enumerate(block_order_list):
        block_trials, cumulative_points = run_block(
            win, stim,
            agent_id          = agent_id,
            block_num         = block_idx + 1,
            n_trials          = args.n_trials,
            cumulative_points = cumulative_points,
            participant_id    = args.participant_id,
            session           = args.session,
            block_order       = block_idx + 1,
            trial_global_offset = global_trial,
            scanner           = args.scanner,
            seed              = args.seed,
        )
        all_trials   += block_trials
        global_trial += len(block_trials)

    # End screen
    show_text(win, stim, INSTRUCTIONS["end"].format(points=cumulative_points))
    win.close()
    core.quit()


if __name__ == "__main__":
    import numpy as np
    main()
