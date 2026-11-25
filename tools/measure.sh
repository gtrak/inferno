#!/usr/bin/env bash
# Produce a timing number that is allowed to be quoted.
#
# Why this exists: I twice reported a figure that did not mean what I said it
# meant. Once it was compile time quoted as parse time, and once it was measured
# on a box that still had `zig`/`test` processes I had orphaned with aborted tool
# calls. Both traps were already written down in my own notes when I walked into
# them, which is proof that notes are not a control - a check you have to remember
# to run is decoration (plan.md 7j).
#
# So the check is mechanical: this script refuses to measure on a busy box, always
# runs the command twice so cold and warm are visible side by side, and fails when
# the two disagree enough that one of them cannot be quoted.
#
# Usage:
#   tools/measure.sh -- <command...>        # default: two runs
#   tools/measure.sh -n 3 -- <command...>   # three runs
#   tools/measure.sh --allow-gpu -- <cmd>   # box may have GPU memory in use
#   tools/measure.sh --budget gate -- <cmd> # compare against tools/time-budgets.tsv
#
# With --budget NAME the run is checked against that budget's expected time and cap.
# Exceeding expected is reported as a finding to diagnose; reaching the cap is an error.
# A budgeted run is capped by `timeout`, so it cannot sit wedged indefinitely.
#
# Quote only the per-run wall times printed here, and say which run you quoted.

set -u -o pipefail

runs=2
allow_gpu=0
budget_name=
while [ $# -gt 0 ]; do
    case "$1" in
        -n)
            runs="$2"
            shift 2
            ;;
        --allow-gpu)
            allow_gpu=1
            shift
            ;;
        --budget)
            budget_name="$2"
            shift 2
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "measure: unexpected argument: $1" >&2
            echo "usage: tools/measure.sh [-n RUNS] [--allow-gpu] -- <command...>" >&2
            exit 2
            ;;
    esac
done

if [ $# -eq 0 ]; then
    echo "measure: no command given" >&2
    echo "usage: tools/measure.sh [-n RUNS] [--allow-gpu] -- <command...>" >&2
    exit 2
fi

if ! [ "$runs" -ge 2 ] 2>/dev/null; then
    # One run cannot distinguish cold from warm, which is the whole failure mode.
    echo "measure: -n must be >= 2 (a single run cannot separate cold from warm)" >&2
    exit 2
fi

# ---- optional budget lookup -------------------------------------------------
budget_expected=""
budget_cap=""
if [ -n "$budget_name" ]; then
    budgets_file="$(dirname "$0")/time-budgets.tsv"
    if ! [ -f "$budgets_file" ]; then
        echo "measure: no budget table at $budgets_file" >&2
        exit 2
    fi
    budget_line="$(awk -F'\t' -v want="$budget_name" \
        '$1 == want {print $2 "\t" $3; found=1} END{if (!found) exit 1}' "$budgets_file")"
    if [ -z "$budget_line" ]; then
        echo "measure: unknown budget '$budget_name'; known budgets:" >&2
        awk -F'\t' '$1 !~ /^#/ && NF >= 3 {print "         " $1 " (expected " $2 "s, cap " $3 "s)"}' \
            "$budgets_file" >&2
        exit 2
    fi
    budget_expected="${budget_line%%	*}"
    budget_cap="${budget_line##*	}"
    echo "measure: budget '$budget_name' expected ${budget_expected}s cap ${budget_cap}s"
fi

# ---- precondition 1: no competing build/test processes ----------------------
# A timing taken next to an orphaned `zig`/`test` is void, and orphaned processes
# are exactly what aborted tool calls leave behind.
competitors=""
for name in zig cargo nvcc test build; do
    found="$(pgrep -x "$name" 2>/dev/null | tr '\n' ' ')"
    if [ -n "$found" ]; then
        competitors="$competitors $name(${found% })"
    fi
done
if [ -n "$competitors" ]; then
    echo "measure: REFUSING - competing processes would invalidate the number:" >&2
    echo "         $competitors" >&2
    echo "         wait for them, or kill them, then measure again." >&2
    exit 3
fi

# ---- precondition 2: the GPUs are not already in use ------------------------
gpu_used=""
if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
        | awk '$1 > 1024 {print $1}' | tr '\n' ' ')"
fi
if [ -n "$gpu_used" ] && [ "$allow_gpu" -eq 0 ]; then
    echo "measure: REFUSING - GPU memory in use (MiB): $gpu_used" >&2
    echo "         another run is holding the card; pass --allow-gpu if that is intended." >&2
    exit 3
fi

# ---- record the environment the number belongs to ---------------------------
load_average="$(cut -d' ' -f1-3 /proc/loadavg)"
disk_free_gib="$(df -BG --output=avail . 2>/dev/null | tail -1 | tr -dc '0-9')"
echo "measure: runs=$runs host-load='$load_average' disk-free=${disk_free_gib:-?}GiB"
if [ -n "$gpu_used" ]; then
    echo "measure: GPU memory in use (MiB): $gpu_used"
fi
echo "measure: command: $*"

# ---- run, timing wall clock per run ----------------------------------------
declare -a seconds=()
index=1
while [ "$index" -le "$runs" ]; do
    started="$(date +%s.%N)"
    if [ -n "$budget_cap" ]; then
        run_status=0
        timeout "$budget_cap" "$@" >/tmp/measure-run-"$index".log 2>&1 || run_status=$?
    else
        run_status=0
        "$@" >/tmp/measure-run-"$index".log 2>&1 || run_status=$?
    fi
    if [ "$run_status" -eq 124 ]; then
        echo "measure: run $index STOPPED at the budget cap (${budget_cap}s) for '$budget_name'." >&2
        echo "         This is not a slow run, it is a wedged or pathological one." >&2
        echo "         Diagnose in this order: orphaned processes, tracked allocator on a" >&2
        echo "         bulk path, cold compile mistaken for the run, then the algorithm." >&2
        tail -5 /tmp/measure-run-"$index".log >&2
        exit 7
    fi
    if [ "$run_status" -ne 0 ]; then
        echo "measure: run $index FAILED - see /tmp/measure-run-$index.log" >&2
        tail -5 /tmp/measure-run-"$index".log >&2
        exit 4
    fi
    finished="$(date +%s.%N)"
    elapsed="$(awk -v a="$started" -v b="$finished" 'BEGIN{printf "%.2f", b-a}')"
    seconds+=("$elapsed")
    echo "measure: run $index = ${elapsed}s (output /tmp/measure-run-$index.log)"
    index=$((index + 1))
done

# ---- the two runs must agree, or neither may be quoted ---------------------
first="${seconds[0]}"
last="${seconds[${#seconds[@]}-1]}"
if awk -v a="$first" -v b="$last" 'BEGIN{exit !(a > 0 && ((b-a)/a > 0.25 || (a-b)/a > 0.25))}'; then
    echo "measure: AMBIGUOUS - first ${first}s vs last ${last}s differ by more than 25%." >&2
    echo "         One of these is cold (build/link/compile) and the other is the run." >&2
    echo "         Do not quote either until you know which; usually the LAST is the run." >&2
    exit 5
fi

echo "measure: quotable = ${last}s (runs agreed within 25%; ${first}s first, ${last}s last)"

if [ -n "$budget_expected" ]; then
    if awk -v got="$last" -v want="$budget_expected" 'BEGIN{exit !(got > want)}'; then
        echo "measure: EXCEEDS EXPECTED - '$budget_name' expected ${budget_expected}s, got ${last}s." >&2
        echo "         Not a reason to wait longer: record the real number in" >&2
        echo "         tools/time-budgets.tsv if it is the new normal, or diagnose why it moved." >&2
        exit 6
    fi
    echo "measure: within budget '$budget_name' (expected ${budget_expected}s, cap ${budget_cap}s)"
fi
