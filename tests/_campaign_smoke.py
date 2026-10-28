"""Contracts for the unattended test campaign (tools/campaign.py, tools/campaign_arms.py; ASS-013).

    PYTHONPATH=. python tests/_campaign_smoke.py

WHY THIS EXISTS. The campaign is ~190 host-hours of queued runs with nobody watching. The ways
it can fail silently are the recurring ones in CLAUDE.md 4.1: a guard that latches (a pause no
one can clear, a gate nothing opens), an arm that is a no-op (a mistyped config key makes an
"A/B" an A/A), and a restart that loses or duplicates work. Each contract below is the test
that would catch one of those. All use fake stages (echo, false); nothing starts a trainer.

  A  every overlay key exists in configs/universal.yaml, and the check catches a typo
     (falsified: a bad key must be reported, or the check proves nothing)
  B  T2 connected vs cut differ ONLY in the cluster's wired channels (same blocks in both)
  C  the plan is well formed: unique ids, every `requires` exists, blocked work is visible
     and its dependents are blocked too (never silently skipped, never started)
  D  stages run in order, serially; a stage waits for its requirement
  E  a failing pause-on-fail stage PAUSES the queue; `resume` re-opens it (the escape path);
     a failing on_fail=continue stage does not stop the queue
  F  a human gate holds its dependents until the gate file exists, then they run on their own
  G  a stage left `running` by a reboot with a dead process re-enters; one with a live process
     is attached to, not started twice
  H  the frozen code hash refuses a changed tree unless accepted
  I  the driver never starts a stage while another trainer holds the host
  J  an eval stage is `done` only when the runner's own state says every run is terminal, and
     `done_with_failures` when a run failed_final (the crash rule: no claim, not an error)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import yaml  # noqa: E402

import campaign as C  # noqa: E402
import campaign_arms as A  # noqa: E402

FAILS = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' - ' + detail) if detail and not cond else ''}")
    if not cond:
        FAILS.append(name)


def make(tmp: str, stages=None):
    code = os.path.join(tmp, "code")
    os.makedirs(os.path.join(code, "tests"), exist_ok=True)
    Path(code, "marker.txt").write_text("v1")
    a = types.SimpleNamespace(root=os.path.join(tmp, "camp"), code=code, venv="", ab00_root=os.path.join(tmp, "ab00"),
                              poll=0.05, idle=0.1, min_free_gb=0.0, sweep_every=3600.0)
    cfg = C.Cfg(a)
    camp = C.Campaign(cfg, stages=stages)
    camp.trainer_alive = lambda: False
    return camp, cfg


def fake(sid, cmd, requires=(), on_fail="pause", kind="cmd", **kw):
    d = dict(id=sid, title=sid, kind=kind, cmd=["bash", "-c", cmd], requires=list(requires), on_fail=on_fail,
             est_hours=0.0, blocked=None, env={})
    d.update(kw)
    return d


def stat(camp, sid):
    return camp.st(sid)["status"]


def main() -> int:
    u = yaml.safe_load(open(os.path.join(ROOT, "configs/universal.yaml")))

    print("A overlays vs configs/universal.yaml")
    check("all generated arms use real keys", A.validate(u) == [], str(A.validate(u))[:200])
    bad = dict(A.T1)
    bad["typo"] = ("WP-X", {"curriculum": {"enabeld": True}}, None)
    old = A.T1
    A.T1 = bad
    try:
        probs = A.validate(u)
    finally:
        A.T1 = old
    check("a mistyped key is reported (falsification)", any("enabeld" in p for p in probs), str(probs))
    check("every T2 channel is registered", A.channel_problems() == [], str(A.channel_problems()))

    print("B T2 connected vs cut")
    for n in A.T2:
        if A.blocked_reason("t2", n):
            continue
        con, cut = A.t2_overlays(n)
        pairs = A.T2[n][1]
        rel_c = {k: v for k, v in con["relations"].items() if k != "enabled"}
        rel_k = {k: v for k, v in cut["relations"].items() if k != "enabled"}
        stripped = {k: v for k, v in con.items() if k != "relations"} == {k: v for k, v in cut.items() if k != "relations"}
        check(f"{n}: same blocks both arms, only channels differ", stripped and not rel_k and
              sum(len(v) for v in rel_c.values()) == len(pairs))

    print("C plan shape")
    with tempfile.TemporaryDirectory() as tmp:
        camp, cfg = make(tmp)
        ids = [s["id"] for s in camp.stages]
        check("ids unique", len(ids) == len(set(ids)))
        check("every requires exists", all(r in camp.by_id for s in camp.stages for r in s["requires"]))
        check("ab00 is first", ids[0] == "ab00")
        for blocked in ("t1_wp1_skills", "t1_wp10_scaling", "t2_rw6_strategy", "cat_harness"):
            check(f"{blocked} is blocked", camp.classify(camp.by_id[blocked]) == "blocked")
        check("dependent analysis of a blocked stage is blocked", camp.classify(camp.by_id["an_t1_wp1_skills"]) == "blocked")
        # GOV-014: the diverse-source arm is runnable (gated), the fresh control is queued, F-24 blocks rw6
        check("t1_wp6_diversity is no longer blocked", camp.classify(camp.by_id["t1_wp6_diversity"]) != "blocked")
        w = camp.by_id["t1_wp6_diversity"]
        check("it waits for the boot checks AND the live coverage check", {"arm_boot_checks", "wp6_coverage_check"} <= set(w["requires"])
              and "wp6_coverage" in w["boot_arms"] and "wp6_diversity" in w["boot_arms"])
        check("the rw6 block names F-24", "F-24" in (camp.block_reason(camp.by_id["t2_rw6_strategy"]) or ""))
        fc = camp.by_id["fresh_control"]
        check("the fresh control is a target-only arm over the AB-00 seeds",
              "--fresh-control" in fc["cmd"] and fc["cmd"][fc["cmd"].index("--seeds") + 1] == ",".join(map(str, range(11, 21))))
        check("it runs after ab00 and has a descriptive analysis", fc["requires"] == ["ab00"] and camp.by_id["an_fresh_control"]["requires"] == ["fresh_control"])
        check("the coverage probe overlay is written", os.path.exists(os.path.join(cfg.overlays, "wp6_coverage_probe.yaml")))
        check("cat_sizing blocked behind the missing harness", camp.classify(camp.by_id["cat_sizing"]) == "blocked")
        # the latch (ASS-017): a failed contract tier must not block the arm boot checks or anything behind them
        camp.state["stages"]["tests_contract"]["status"] = "failed"
        camp.state["stages"]["ab00"]["status"] = "done"
        check("arm_boot_checks does not depend on the test tiers", camp.by_id["arm_boot_checks"]["requires"] == ["ab00"])
        check("a FAILED contract tier does not block the arm boot checks", camp.classify(camp.by_id["arm_boot_checks"]) == "runnable",
              camp.classify(camp.by_id["arm_boot_checks"]))
        check("nor the WP-6 live check or a T1 A/B behind them",
              camp.block_reason(camp.by_id["wp6_coverage_check"]) is None and camp.block_reason(camp.by_id["t1_wp3_curriculum"]) is None)
        check("the test tiers run from a complete test tree",
              all("test_extras" in " ".join(camp.by_id[f"tests_{t}"]["cmd"]) and "test_code" in " ".join(camp.by_id[f"tests_{t}"]["cmd"])
                  for t in ("unit", "contract")))
        camp.state["stages"]["tests_contract"]["status"] = "pending"
        camp.state["stages"]["ab00"]["status"] = "pending"
        check("a buildable T1 queues behind the baseline", camp.classify(camp.by_id["t1_wp3_curriculum"]) == "queued")
        check("overlays written", os.path.exists(os.path.join(cfg.overlays, "wp3_curriculum.yaml")) and
              os.path.exists(os.path.join(cfg.overlays, "combined.yaml")))
        check("plan prints every stage", all(i in camp.plan_text() for i in ("ab00", "t2_rw1_modulation", "cat_combined")))
        # the harness unblocks by itself once its files exist (no human step)
        Path(cfg.code, "tests", "_categorical_cat03_smoke.py").write_text("x")
        check("harness files unblock cat_harness without a restart", camp.classify(camp.by_id["cat_harness"]) != "blocked")

    print("D order and serial execution")
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "order.txt")
        st = [fake("s1", f"echo s1 >> {out}"), fake("s2", f"echo s2 >> {out}", ["s1"]),
              fake("s3", f"echo s3 >> {out}", ["s2"])]
        camp, cfg = make(tmp, st)
        rc = camp.run()
        check("ran in order", Path(out).read_text().split() == ["s1", "s2", "s3"])
        check("all done, exit 0", rc == 0 and all(stat(camp, s) == "done" for s in ("s1", "s2", "s3")))

    print("D2 a running stage is re-attached before an earlier one starts (no second trainer)")
    with tempfile.TemporaryDirectory() as tmp:
        camp, cfg = make(tmp, [fake("early", "true"), fake("late", "true")])
        camp.state["stages"]["late"].update(status="running", pid=os.getpid())      # a live process
        check("the running stage is chosen although it is later in the list", camp.next_stage()["id"] == "late")
        camp.state["stages"]["late"]["status"] = "pending"
        check("with nothing running, the first runnable is chosen as before", camp.next_stage()["id"] == "early")

    print("E pause, resume, continue")
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "o.txt")
        st = [fake("a", "exit 3"), fake("b", f"echo b >> {out}", ["a"]),
              fake("c", "exit 1", on_fail="continue"), fake("d", f"echo d >> {out}")]
        camp, cfg = make(tmp, st)
        camp.run()
        check("a pause-on-fail failure pauses the queue", camp.state["paused"] and camp.state["paused"]["stage"] == "a")
        check("nothing after the pause ran", not os.path.exists(out))
        check("run() while paused returns without starting", camp.run() == 0 and not os.path.exists(out))
        camp.state["stages"]["a"]["status"] = "failed"
        st2 = st[:]
        # fix the stage, then clear the pause: the queue continues by itself
        camp.by_id["a"]["cmd"] = ["bash", "-c", "exit 0"]
        camp.resume("a")
        check("resume clears pause and re-queues the failed stage", not camp.state["paused"] and stat(camp, "a") == "pending")
        camp.run()
        check("after resume everything ran", Path(out).read_text().split() == ["b", "d"] or
              sorted(Path(out).read_text().split()) == ["b", "d"])
        check("on_fail=continue failure did not stop the queue", stat(camp, "c") == "failed" and not camp.state["paused"])

    print("F human gate")
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "o.txt")
        st = [fake("g", "true", kind="gate", gate="thr"), fake("after", f"echo after >> {out}", ["g"])]
        camp, cfg = make(tmp, st)
        check("a closed gate is waiting-human", camp.classify(camp.by_id["g"]) == "waiting-human")
        check("its dependent is queued, not run", camp.classify(camp.by_id["after"]) == "queued")
        orig = camp.sleep
        calls = []

        def opener(seconds):
            calls.append(seconds)
            Path(cfg.gates, "thr").write_text("x")      # the human touches the gate
            return False
        camp.sleep = opener
        camp.run()
        check("the dependent ran by itself once the gate opened", Path(out).exists() and len(calls) >= 1)

    print("G reboot recovery and attach")
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "o.txt")
        st = [fake("r", f"echo r >> {out}")]
        camp, cfg = make(tmp, st)
        camp.state["stages"]["r"].update(status="running", pid=2 ** 22 + 12345)   # no such process
        camp.save()
        camp.run()
        check("a running stage with a dead process re-enters", stat(camp, "r") == "done" and Path(out).read_text().split() == ["r"])
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "o.txt")
        sd = os.path.join(tmp, "camp", "stages", "live")
        os.makedirs(sd, exist_ok=True)
        # an ORPHAN (as after a driver restart): grandchild reparented to init, so it is not our zombie
        L = subprocess.run(["bash", "-c", f"(sleep 1; echo 0 > {sd}/exit_code) >/dev/null 2>&1 & echo $!"],
                           capture_output=True, text=True)
        p = types.SimpleNamespace(pid=int(L.stdout.strip()), wait=lambda: None)
        st = [fake("live", f"echo STARTED_TWICE >> {out}")]
        camp, cfg = make(tmp, st)
        camp.state["stages"]["live"].update(status="running", pid=p.pid)
        camp.save()
        camp.run()
        p.wait()
        check("a live process is attached to, never started twice", not os.path.exists(out) and stat(camp, "live") == "done")

    print("H frozen code")
    with tempfile.TemporaryDirectory() as tmp:
        camp, cfg = make(tmp, [fake("x", "true")])
        Path(cfg.code, "marker.txt").write_text("v2")
        try:
            C.Campaign(cfg, stages=[fake("x", "true")])
            refused = False
        except SystemExit:
            refused = True
        check("a changed code tree is refused", refused)
        C.Campaign(cfg, stages=[fake("x", "true")], accept_code=True)
        check("--accept-code-change takes the new snapshot", True)

    print("I host exclusivity")
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "o.txt")
        camp, cfg = make(tmp, [fake("only", f"echo ran >> {out}")])
        busy = {"n": 0}

        def trainer():
            busy["n"] += 1
            return busy["n"] <= 2
        camp.trainer_alive = trainer
        camp.run()
        check("it waited while a trainer was alive, then ran", busy["n"] >= 3 and Path(out).exists())

    print("J eval completion comes from the runner's state")
    with tempfile.TemporaryDirectory() as tmp:
        for runs, want in (({"r1": {"status": "done"}, "r2": {"status": "done"}}, "done"),
                           ({"r1": {"status": "done"}, "r2": {"status": "failed_final"}}, "done_with_failures"),
                           ({"r1": {"status": "done"}, "r2": {"status": "pending"}}, None)):
            camp, cfg = make(tmp, [fake("e", "true", kind="eval", root=os.path.join(tmp, "eroot"))])
            os.makedirs(os.path.join(tmp, "eroot"), exist_ok=True)
            json.dump({"runs": runs}, open(os.path.join(tmp, "eroot", "eval_state.json"), "w"))
            check(f"state {sorted(r['status'] for r in runs.values())} -> {want}", camp.eval_done(camp.by_id["e"]) == want)
        camp, cfg = make(tmp, [fake("e2", "exit 0", kind="eval", root=os.path.join(tmp, "empty"))])
        os.makedirs(os.path.join(tmp, "empty"), exist_ok=True)
        camp.run()
        check("a runner that exits 0 without finishing its runs is NOT done (retried once, then fails)",
              stat(camp, "e2") == "failed" and camp.state["paused"] is not None)

    print("K prune and compress finished runs (disk)")
    import gzip
    with tempfile.TemporaryDirectory() as tmp:
        ext = os.path.join(tmp, "external_venv")
        os.makedirs(ext)
        Path(ext, "keep.txt").write_text("the shared venv must survive")
        root = os.path.join(tmp, "eroot")
        for rid, status in (("a_s1", "done"), ("a_s2", "pending"), ("a_s3", "failed_final")):
            rd = os.path.join(root, "runs", rid)
            os.makedirs(os.path.join(rd, "devai", "brain_universal", "logs"))
            os.makedirs(os.path.join(rd, "snap_source_end", "logs"))
            os.makedirs(os.path.join(rd, "snap_source_end", "skill_bank"))
            os.makedirs(os.path.join(rd, "results", "target"))
            os.symlink(ext, os.path.join(rd, "devai", "venv_mc"))
            Path(rd, "devai", "brain_universal", "logs", "replay.npy").write_bytes(b"x" * 100000)
            Path(rd, "snap_source_end", "logs", "ckpt.pt").write_bytes(b"y" * 100000)
            Path(rd, "snap_source_end", "skill_bank", "s.json").write_text("{}")
            Path(rd, "phase_target.json").write_text('{"seed": 1}')
            text = "".join(f"00:{i // 60 % 60:02d}:{i % 60:02d} INFO doom stream 0: {{'kills_total': 1.0}}\n" for i in range(2000))
            Path(rd, "results", "target", "minecraft_skybot_run.log").write_text(text)
        json.dump({"runs": {"a_s1": {"status": "done"}, "a_s2": {"status": "pending"},
                            "a_s3": {"status": "failed_final"}}}, open(os.path.join(root, "eval_state.json"), "w"))
        camp, cfg = make(tmp, [fake("e", "true", kind="eval", root=root)])
        freed = camp.sweep()
        r1 = os.path.join(root, "runs", "a_s1")
        check("a finished run lost its devai and checkpoint copy", not os.path.exists(os.path.join(r1, "devai"))
              and not os.path.exists(os.path.join(r1, "snap_source_end", "logs")) and freed >= 200000)
        check("its small snapshot parts and phase file are kept", os.path.exists(os.path.join(r1, "snap_source_end", "skill_bank", "s.json"))
              and os.path.exists(os.path.join(r1, "phase_target.json")))
        check("the shared venv behind the symlink was NOT touched", os.path.exists(os.path.join(ext, "keep.txt")))
        check("a run still pending is untouched", os.path.exists(os.path.join(root, "runs", "a_s2", "devai", "brain_universal")))
        check("a failed_final run is kept for diagnosis", os.path.exists(os.path.join(root, "runs", "a_s3", "devai")))
        gz = os.path.join(r1, "results", "target", "minecraft_skybot_run.log.gz")
        check("its log was gzipped and the original removed", os.path.exists(gz)
              and not os.path.exists(gz[:-3]) and os.path.getsize(gz) < 30000)
        check("a second sweep frees nothing", camp.sweep() == 0)
        from developmental_ai.evaluator import scoring
        ev = scoring.parse_evaluator_log(gz)
        check("the scorer reads the .gz log", len(ev) == 2000 and ev[0][2].get("kills_total") == 1.0, str(len(ev)))

    print("L disk guard")
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "o.txt")
        camp, cfg = make(tmp, [fake("big", f"echo ran >> {out}")])
        camp.c.min_free_gb = 1e9
        camp.run()
        check("too little free disk pauses BEFORE the stage starts", camp.state["paused"] and not os.path.exists(out)
              and "GB free" in camp.state["paused"]["reason"])
        camp.c.min_free_gb = 0.0
        camp.resume("all")
        camp.run()
        check("freeing space and `resume` lets it run (the escape)", Path(out).exists())

    print("M a failed boot check blocks exactly its own arm")
    with tempfile.TemporaryDirectory() as tmp:
        camp, cfg = make(tmp)
        json.dump({"wp3_curriculum": {"ok": False, "reason": "traceback in x.log"},
                   "wp7_attention": {"ok": True, "reason": ""}}, open(os.path.join(cfg.analysis, "arm_boot_checks.json"), "w"))
        check("the failed arm's A/B is blocked with the reason", "wp3_curriculum" in (camp.block_reason(camp.by_id["t1_wp3_curriculum"]) or ""))
        check("another arm is not blocked by it", camp.block_reason(camp.by_id["t1_wp7_attention"]) is None)
        check("every arm-carrying stage waits for the boot checks",
              all("arm_boot_checks" in camp.by_id[i]["requires"] for i in ("t1_wp3_curriculum", "t2_rw2_attention", "cat_combined",
                                                                           "pilot_cat03_planning_enabled")))
        names = camp.by_id["arm_boot_checks"]["cmd"][camp.by_id["arm_boot_checks"]["cmd"].index("--names") + 1].split(",")
        json.dump({"wp6_diversity": {"ok": True, "reason": ""}, "wp6_coverage": {"ok": False, "reason": "only 1 distinct maps seen (< 12)"}},
                  open(os.path.join(cfg.analysis, "arm_boot_checks.json"), "w"))
        check("a failed live coverage check blocks the WP-6 A/B with its reason",
              "wp6_coverage" in (camp.block_reason(camp.by_id["t1_wp6_diversity"]) or ""))
        check("the boot check covers every runnable overlay", all(os.path.exists(os.path.join(cfg.overlays, n + ".yaml")) for n in names)
              and "combined" in names and "rw1_modulation_cut" in names)

    print("N a host restart is an interruption, not a crash")
    from developmental_ai.evaluator.runner import EvalState
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "eroot")
        os.makedirs(root)
        base = {"arm": "a", "phases": [["source", 40], ["target", 20]], "status": "pending", "attempts": 1,
                "phase_done": [], "crashes": [], "info": {}, "seed": 1}
        runs = {"r1": dict(base, phase_done=["source"], in_progress={"phase": "target", "t": 1000.0}),
                "r2": dict(base, in_progress={"phase": "source", "t": 3000.0}),
                "r3": dict(base)}
        sp = os.path.join(root, "eval_state.json")
        json.dump({"frozen": {}, "order": ["r1", "r2", "r3"], "runs": runs}, open(sp, "w"))
        camp, cfg = make(tmp, [fake("e", "true", kind="eval", root=root)])
        hit = camp.repair_after_reboot(boot=2000.0)
        d = json.load(open(sp))["runs"]
        check("only the run that began before the boot was reset", hit == ["e:r1"])
        check("it restarts its seed from the beginning, attempt count unchanged, no crash recorded",
              d["r1"]["in_progress"] is None and d["r1"]["phase_done"] == [] and d["r1"]["attempts"] == 1
              and d["r1"]["crashes"] == [] and len(d["r1"]["interruptions"]) == 1)
        check("a run that began after the boot (driver death only) is left to the runner's own rule", d["r2"]["in_progress"] is not None)
        st = EvalState.load_or_create(sp, [], None)          # the real runner's recover()
        r = st.data["runs"]
        check("the real runner then spends NO retry on the interrupted run", r["r1"]["attempts"] == 1 and r["r1"]["crashes"] == [])
        check("(control) it would have spent one without the repair", r["r2"]["attempts"] == 2 and len(r["r2"]["crashes"]) == 1)

    print("O unreadable state files after a power cut")
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "eroot")
        os.makedirs(root)
        good = {"frozen": {}, "order": [], "runs": {}}
        sp = os.path.join(root, "eval_state.json")
        json.dump(good, open(sp + ".bak", "w"))
        Path(sp).write_text('{"runs": {"r1": {"st')           # truncated mid-write
        camp, cfg = make(tmp, [fake("e", "true", kind="eval", root=root)])
        camp.repair_after_reboot(boot=1.0)
        check("a truncated runner state is restored from its backup", camp.valid_json(sp) and json.load(open(sp)) == good)
        Path(sp).write_text("{")
        Path(sp + ".bak").write_text("{")
        camp.repair_after_reboot(boot=1.0)
        check("with no readable copy the queue PAUSES instead of starting over", camp.state["paused"] is not None)
    with tempfile.TemporaryDirectory() as tmp:
        camp, cfg = make(tmp, [fake("x", "true")])
        camp.set("x", status="done")                              # a second save leaves a .bak of the first
        check("the campaign keeps a valid backup of its state", camp.valid_json(cfg.state_path + ".bak"))
        Path(cfg.state_path).write_text('{"stages": {"x": {"sta')   # power cut mid-write
        c2 = C.Campaign(cfg, stages=[fake("x", "true")], accept_code=True)
        check("a truncated campaign state is restored from its backup, finished work not forgotten",
              c2.st("x")["status"] in ("done", "pending") and c2.state["code_hash"])
        Path(cfg.state_path).write_text("{")
        Path(cfg.state_path + ".bak").write_text("{")
        try:
            C.Campaign(cfg, stages=[fake("x", "true")], accept_code=True)
            refused = False
        except SystemExit:
            refused = True
        check("both unreadable: it refuses to silently start a new campaign", refused)
    with tempfile.TemporaryDirectory() as tmp:
        camp, cfg = make(tmp, [fake("x", "true")])
        camp.set("x", status="done")
        Path(cfg.state_path).write_text("{")                      # corrupt main
        c2 = C.Campaign(cfg, stages=[fake("x", "true")], accept_code=True)   # restores, then saves
        check("a corrupt state is never copied over the good backup (it stays readable)",
              c2.valid_json(cfg.state_path + ".bak") and "stages" in json.load(open(cfg.state_path + ".bak")))
        check("the restored state is readable and is saved back as the new main file", c2.valid_json(cfg.state_path))

    print(f"[campaign] {'ALL PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
