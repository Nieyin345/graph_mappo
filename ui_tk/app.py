
"""QKD-RL Training Desktop App - 4 tabs."""
from __future__ import annotations
import json, os, sys, threading, time
from pathlib import Path
from tkinter import ttk, messagebox, filedialog
import tkinter as tk
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from ui_tk.utils.config_manager import get_profile_keys, list_checkpoints, get_baselines, load_baselines_config, save_baselines_config
from ui_tk.utils.train_manager import TrainProcess, generate_command, register_train_process


class QKDRLApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("QKD-RL Training Manager")
        self.root.geometry("1280x800")
        self.train_proc = None
        self._monitor_job = None
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=6, pady=6)
        self.tab_train = ttk.Frame(self.notebook)
        self.tab_viz = ttk.Frame(self.notebook)
        self.tab_eval = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_train, text="  Train  ")
        self.notebook.add(self.tab_viz, text="  Viz  ")
        self.notebook.add(self.tab_eval, text="  Eval  ")
        self._build_train_tab()
        self._build_viz_tab()
        self._build_eval_tab()
        self._load_global_config()
        self._load_mode_config()
        self._update_eval_total()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    # ===== TAB 1: Training =====
    def _load_global_config(self):
        p = ROOT / "configs" / "global.yaml"
        if not p.is_file(): return
        try:
            c = yaml.safe_load(open(p, encoding="utf-8")) or {}
            g = c.get("global", {}); t = g.get("training", {}); w = t.get("window", {})
            if "start_day" in w: self.g_ts.set(str(w["start_day"]))
            if "end_day" in w: self.g_te.set(str(w["end_day"]))
            if "request_seed" in t: self.g_rs.set(str(t["request_seed"]))
            v = g.get("validation", {}); vw = v.get("window", {})
            if "start_day" in vw: self.g_vs.set(str(vw["start_day"]))
            if "end_day" in vw: self.g_ve.set(str(vw["end_day"]))
            if "request_seeds" in v: self.g_vse.set(",".join(str(s) for s in v["request_seeds"]))
            if "episodes" in v: self.g_vep.set(str(v["episodes"]))
            if "episode_steps" in v: self.g_vd.set(str(v["episode_steps"]))
            elif "episode_days" in v: self.g_vd.set(str(int(v["episode_days"]) * 1440))
            # Update eval tab global info
            wd = v.get("window", {})
            seeds = v.get("request_seeds", [7])
            esteps = v.get("episode_steps", int(v.get("episode_days", 1)) * 1440)
            self.e_global_info.configure(text=f"验证窗口: {wd.get('start_day','?')}~{wd.get('end_day','?')} 天, 种子: {seeds}, 步长/局: {esteps}")
        except Exception as e: self.g_st.configure(text=f"load err: {e}", foreground="red")

    def _save_global(self):
        try:
            cfg = {"global": {"training": {"window": {"start_day": int(self.g_ts.get()), "end_day": int(self.g_te.get())},
                "request_seed": int(self.g_rs.get())},
                "validation": {"window": {"start_day": int(self.g_vs.get()), "end_day": int(self.g_ve.get())},
                "request_seeds": [int(s.strip()) for s in self.g_vse.get().split(",") if s.strip()],
                "episodes": int(self.g_vep.get()), "episode_steps": int(self.g_vd.get())}}}
            yaml.safe_dump(cfg, open(ROOT/"configs"/"global.yaml", "w", encoding="utf-8"), sort_keys=False, allow_unicode=True, indent=2)
            self.g_st.configure(text="saved", foreground="green")
            self.root.after(2000, lambda: self.g_st.configure(text=""))
        except Exception as e: self.g_st.configure(text=f"save err: {e}", foreground="red")

    # ===== TAB 2: Training =====
    def _build_train_tab(self):
        paned = ttk.PanedWindow(self.tab_train, orient="horizontal")
        paned.pack(fill="both", expand=True)
        left = ttk.Frame(paned, width=450); paned.add(left, weight=0)
        right = ttk.Frame(paned); paned.add(right, weight=1)
        self._build_train_config(left); self._build_monitor(right)

    def _build_train_config(self, parent):
        cv = tk.Canvas(parent, width=430)
        sb = ttk.Scrollbar(parent, orient="vertical", command=cv.yview)
        sf = ttk.Frame(cv)
        sf.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))
        cv.create_window((0, 0), window=sf, anchor="nw"); cv.configure(yscrollcommand=sb.set)
        r = 0
        ttk.Label(sf, text="Mode", font=("", 10, "bold")).grid(row=r, column=0, columnspan=2, sticky="w", pady=(6,2)); r+=1
        ttk.Label(sf, text="mode:").grid(row=r, column=0, sticky="w", padx=5, pady=2)
        self.combo_mode = ttk.Combobox(sf, values=get_profile_keys(), state="readonly", width=26)
        self.combo_mode.set("fixed_episode")
        self.combo_mode.bind("<<ComboboxSelected>>", self._on_mode_change)
        self.combo_mode.grid(row=r, column=1, sticky="ew", padx=5, pady=2); r+=1
        ttk.Label(sf, text="run name:").grid(row=r, column=0, sticky="w", padx=5, pady=2)
        self.en_name = ttk.Entry(sf, width=28)
        self.en_name.insert(0, f"exp_{int(time.time())}")
        self.en_name.grid(row=r, column=1, sticky="ew", padx=5, pady=2); r+=1
        ttk.Label(sf, text="checkpoint:").grid(row=r, column=0, sticky="w", padx=5, pady=2)
        cf = ttk.Frame(sf); cf.grid(row=r, column=1, sticky="ew", padx=5, pady=2)
        self.ckpt_var = tk.StringVar(value="")
        ttk.Label(cf, textvariable=self.ckpt_var, foreground="gray", width=22).pack(side="left")
        ttk.Button(cf, text="browse", command=self._browse_ckpt, width=6).pack(side="right"); r+=1
        ttk.Separator(sf, orient="horizontal").grid(row=r, column=0, columnspan=2, sticky="ew", pady=6); r+=1
        ttk.Label(sf, text="Parameters", font=("", 10, "bold")).grid(row=r, column=0, columnspan=2, sticky="w", pady=(2,2)); r+=1
        self.pv = {}
        for key, defv in [("num_updates","400"),("rollout_steps","240"),("episodes_per_update","4"),
            ("ppo_epochs","1"),("minibatch_size","256"),("entropy_coef","0.001"),
            ("clip_eps","0.2"),("gamma","0.99"),("actor_lr","0.0003"),("critic_lr","0.0003"),
            ("eval_episodes","2"),("eval_steps","1440"),("eval_interval","10"),("checkpoint_interval","10"),
            ("temp_start","1.0"),("temp_end","1.0"),("temp_updates","1000")]:
            ttk.Label(sf, text=f"{key}:").grid(row=r, column=0, sticky="w", padx=5, pady=1)
            self.pv[key] = tk.StringVar(value=defv)
            ttk.Entry(sf, textvariable=self.pv[key], width=12).grid(row=r, column=1, sticky="w", padx=5, pady=1); r+=1
        ttk.Label(sf, text="(eval_steps=0用模式默认步长)", foreground="gray").grid(row=r, column=1, sticky="w", padx=5); r+=1
        ttk.Separator(sf, orient="horizontal").grid(row=r, column=0, columnspan=2, sticky="ew", pady=6); r+=1
        self.hist_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(sf, text="History Encoder", variable=self.hist_var).grid(row=r, column=0, columnspan=2, sticky="w", padx=5); r+=1
        # Global config fields
        ttk.Separator(sf, orient="horizontal").grid(row=r, column=0, columnspan=2, sticky="ew", pady=6); r+=1
        ttk.Label(sf, text="Global Config", font=("", 10, "bold")).grid(row=r, column=0, columnspan=2, sticky="w", pady=(2,2)); r+=1
        self.g_ts = tk.StringVar(value="0"); self.g_te = tk.StringVar(value="295"); self.g_rs = tk.StringVar(value="7")
        self.g_vs = tk.StringVar(value="330"); self.g_ve = tk.StringVar(value="365"); self.g_vse = tk.StringVar(value="7")
        self.g_vep = tk.StringVar(value="1"); self.g_vd = tk.StringVar(value="2")
        ttk.Label(sf, text="train_start:").grid(row=r, column=0, sticky="w", padx=5, pady=1)
        ttk.Entry(sf, textvariable=self.g_ts, width=10).grid(row=r, column=1, sticky="w", padx=5, pady=1); r+=1
        ttk.Label(sf, text="train_end:").grid(row=r, column=0, sticky="w", padx=5, pady=1)
        ttk.Entry(sf, textvariable=self.g_te, width=10).grid(row=r, column=1, sticky="w", padx=5, pady=1); r+=1
        ttk.Label(sf, text="req_seed:").grid(row=r, column=0, sticky="w", padx=5, pady=1)
        ttk.Entry(sf, textvariable=self.g_rs, width=10).grid(row=r, column=1, sticky="w", padx=5, pady=1); r+=1
        ttk.Label(sf, text="val_start:").grid(row=r, column=0, sticky="w", padx=5, pady=1)
        ttk.Entry(sf, textvariable=self.g_vs, width=10).grid(row=r, column=1, sticky="w", padx=5, pady=1); r+=1
        ttk.Label(sf, text="val_end:").grid(row=r, column=0, sticky="w", padx=5, pady=1)
        ttk.Entry(sf, textvariable=self.g_ve, width=10).grid(row=r, column=1, sticky="w", padx=5, pady=1); r+=1
        ttk.Label(sf, text="val_seeds:").grid(row=r, column=0, sticky="w", padx=5, pady=1)
        ttk.Entry(sf, textvariable=self.g_vse, width=10).grid(row=r, column=1, sticky="w", padx=5, pady=1); r+=1
        ttk.Label(sf, text="局步长(episode_steps):").grid(row=r, column=0, sticky="w", padx=5, pady=1)
        ttk.Entry(sf, textvariable=self.g_vd, width=10).grid(row=r, column=1, sticky="w", padx=5, pady=1); r+=1
        self.g_st = ttk.Label(sf, text="", foreground="green")
        self.g_st.grid(row=r, column=0, columnspan=2, sticky="w", padx=5); r+=1
        ttk.Button(sf, text="Save Global", command=self._save_global, width=25).grid(row=r, column=0, columnspan=2, pady=2); r+=1
        r+=1; self.t_st = ttk.Label(sf, text="", foreground="green")
        self.t_st.grid(row=r, column=0, columnspan=2, sticky="w", padx=5); r+=1
        ttk.Button(sf, text="Save Mode Config", command=self._save_mode, width=25).grid(row=r, column=0, columnspan=2, pady=2); r+=1
        self.btn_launch = ttk.Button(sf, text="Launch Training", command=self._launch, width=25)
        self.btn_launch.grid(row=r, column=0, columnspan=2, pady=4); r+=1
        self.btn_stop = ttk.Button(sf, text="Stop Training", command=self._stop, width=25, state="disabled")
        self.btn_stop.grid(row=r, column=0, columnspan=2, pady=2); r+=1
        self.lbl_st = ttk.Label(sf, text="idle", foreground="gray")
        self.lbl_st.grid(row=r, column=0, columnspan=2, sticky="w", padx=5, pady=4); r+=1
        sf.grid_rowconfigure(r, weight=1); sf.columnconfigure(1, weight=1)
        cv.pack(side="left", fill="both", expand=True); sb.pack(side="right", fill="y")
        cv.bind_all("<MouseWheel>", lambda e: cv.yview_scroll(int(-1*(e.delta/120)), "units"))

    def _load_mode_config(self):
        try:
            profile = self.combo_mode.get(); p = ROOT/"configs"/"modes"/f"{profile}.yaml"
            if not p.is_file(): return
            c = yaml.safe_load(open(p, encoding="utf-8")) or {}
            tr = c.get("train", {})
            for k in ["num_updates","rollout_steps","episodes_per_update","gamma"]:
                if k in tr: self.pv[k].set(str(tr[k]))
            pp = tr.get("ppo", {})
            if "epochs" in pp: self.pv["ppo_epochs"].set(str(pp["epochs"]))
            if "minibatch_size" in pp: self.pv["minibatch_size"].set(str(pp["minibatch_size"]))
            if "entropy_coef" in pp: self.pv["entropy_coef"].set(str(pp["entropy_coef"]))
            if "clip_eps" in pp: self.pv["clip_eps"].set(str(pp["clip_eps"]))
            opt = tr.get("optimizer", {})
            if "actor_lr" in opt: self.pv["actor_lr"].set(str(opt["actor_lr"]))
            if "critic_lr" in opt: self.pv["critic_lr"].set(str(opt["critic_lr"]))
            lg = tr.get("logging", {})
            if "eval_episodes" in lg: self.pv["eval_episodes"].set(str(lg["eval_episodes"]))
            if "eval_steps" in lg: self.pv["eval_steps"].set(str(lg["eval_steps"]))
            if "eval_interval" in lg: self.pv["eval_interval"].set(str(lg["eval_interval"]))
            if "checkpoint_interval" in lg: self.pv["checkpoint_interval"].set(str(lg["checkpoint_interval"]))
            ts = tr.get("temperature_schedule", {})
            if "start" in ts: self.pv["temp_start"].set(str(ts["start"]))
            if "end" in ts: self.pv["temp_end"].set(str(ts["end"]))
            if "updates" in ts: self.pv["temp_updates"].set(str(ts["updates"]))
            env = c.get("env", {})
            if "episode_steps" in env: self.pv["rollout_steps"].set(str(env["episode_steps"]))
            feats = c.get("features", {})
            self.hist_var.set(feats.get("history_encoder", {}).get("enabled", True))
        except Exception as e: self.t_st.configure(text=f"load err: {e}", foreground="red")

    def _save_mode(self):
        try:
            profile = self.combo_mode.get()
            cfg = {
                "env": {"episode_steps": int(self.pv["rollout_steps"].get()), "episode_start_mode": "random_day", "day_steps": 1440},
                "train": {"num_updates": int(self.pv["num_updates"].get()), "rollout_steps": int(self.pv["rollout_steps"].get()),
                    "episodes_per_update": int(self.pv["episodes_per_update"].get()), "episode_steps_fixed": True,
                    "n_rollout_workers": 1, "rollout_batch": True, "gamma": float(self.pv["gamma"].get()),
                    "gae_lambda": 0.95, "value_target": "gae",
                    "ppo": {"epochs": int(self.pv["ppo_epochs"].get()), "minibatch_size": int(self.pv["minibatch_size"].get()),
                        "batch_chunk": min(512, int(self.pv["minibatch_size"].get())),
                        "entropy_coef": float(self.pv["entropy_coef"].get()), "clip_eps": float(self.pv["clip_eps"].get()),
                        "normalize_advantages": True, "value_coef": 0.5, "max_grad_norm": 0.5, "target_kl": 0.03},
                    "optimizer": {"type": "adam", "actor_lr": float(self.pv["actor_lr"].get()), "critic_lr": float(self.pv["critic_lr"].get())},
                    "logging": {"log_interval": 1,
                        "checkpoint_interval": int(self.pv["checkpoint_interval"].get()),
                        "eval_interval": int(self.pv["eval_interval"].get()),
                        "eval_episodes": max(1, int(self.pv["eval_episodes"].get())),
                        "eval_steps": max(0, int(self.pv["eval_steps"].get()))},
                    "temperature_schedule": {
                        "start": float(self.pv["temp_start"].get()),
                        "end": float(self.pv["temp_end"].get()),
                        "updates": max(1, int(self.pv["temp_updates"].get()))}}}
            if self.hist_var.get():
                cfg["features"] = {"history_encoder": {"enabled": True, "hidden_dim": 64, "seq_len": 240,
                    "node": {"include_arrived": False, "include_served": False, "include_failed": False, "include_qkp_total": False},
                    "physical_edge": {"include_qkp_level": False, "include_available": False, "include_activated": False},
                    "demand_edge": {"include_pending_wait_buckets": True}}}
            else: cfg["features"] = {"history_encoder": {"enabled": False}}
            yaml.safe_dump(cfg, open(ROOT/"configs"/"modes"/f"{profile}.yaml", "w", encoding="utf-8"), sort_keys=False, allow_unicode=True, indent=2)
            self.t_st.configure(text="saved", foreground="green")
            self.root.after(2000, lambda: self.t_st.configure(text=""))
        except Exception as e: self.t_st.configure(text=f"save err: {e}", foreground="red")

    def _on_mode_change(self, e=None): self._load_mode_config()
    def _browse_ckpt(self):
        p = filedialog.askopenfilename(title="select checkpoint", initialdir=str(ROOT/"outputs"), filetypes=[("pt","*.pt"),("*","*.*")])
        if p: self.ckpt_var.set(p)

    def _launch(self):
        try:
            name = self.en_name.get().strip()
            if not name: messagebox.showerror("err", "enter run name"); return
            self._save_mode()
            profile = self.combo_mode.get(); ckpt = self.ckpt_var.get().strip() or None
            known = {"random_episode", "continuous", "fixed_day", "curriculum", "demand_edge"}
            mode = profile if profile in known else "random_episode"
            cmd = generate_command(mode, name, config_files=[f"modes/{profile}.yaml"], checkpoint=ckpt)
            od = ROOT/"outputs"/name; od.mkdir(parents=True, exist_ok=True)
            self.train_proc = TrainProcess(name, od); register_train_process(name, self.train_proc)
            self._clear_console(); self._append_console(f"start: {name}")
            self.train_proc.start(cmd); self._update_ui(True); self._start_monitor()
        except Exception as e: messagebox.showerror("err", str(e))

    def _stop(self):
        if self.train_proc: self.train_proc.stop(); self._append_console("stopped"); self._update_ui(False)

    def _build_monitor(self, parent):
        ttk.Label(parent, text="Console", font=("",10,"bold")).pack(anchor="w", pady=(4,0))
        self.tc = tk.Text(parent, height=12, wrap="word", state="disabled", bg="#1e1e1e", fg="#d4d4d4", font=("Consolas",9))
        self.tc.pack(fill="both", expand=False, padx=4, pady=2)
        ttk.Label(parent, text="Metrics", font=("",10,"bold")).pack(anchor="w", pady=(8,0))
        mf = ttk.Frame(parent); mf.pack(fill="x")
        self.ml = {}
        for i,(l,k) in enumerate([("Update","up"),("SR","sr"),("Reward","rw")]):
            ttk.Label(mf, text=l, font=("",8)).grid(row=0,column=i,padx=10)
            self.ml[k] = ttk.Label(mf, text="--", font=("",12,"bold"))
            self.ml[k].grid(row=1,column=i,padx=10)
        self.fig = Figure(figsize=(6,3), dpi=80)
        self.ax = self.fig.add_subplot(111); self.ax.grid(True, alpha=0.3); self.fig.tight_layout()
        self.canv = FigureCanvasTkAgg(self.fig, master=parent)
        self.canv.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)

    def _start_monitor(self):
        if not self.train_proc: return
        self.train_proc.poll()
        con = self.train_proc.console
        if con:
            for line in con.split("\n")[-3:]:
                if line.strip(): self._append_console(line)
        m = self.train_proc.metrics
        if m:
            last = m[-1]
            self.ml["up"].configure(text=str(last.get("update","?")))
            self.ml["sr"].configure(text=f"{last.get('mean_success_rate',0):.3f}")
            self.ml["rw"].configure(text=f"{last.get('mean_reward',0):.0f}")
            ups = [x["update"] for x in m if "mean_success_rate" in x]
            vls = [x["mean_success_rate"] for x in m if "mean_success_rate" in x]
            if len(ups) > 1:
                self.ax.clear(); self.ax.plot(ups, vls, "g-", lw=1.5)
                self.ax.grid(True, alpha=0.3); self.fig.tight_layout(); self.canv.draw()
        s = self.train_proc.status
        if s == "running": self.lbl_st.configure(text=f"running ({len(m)})", foreground="green")
        elif s == "done": self.lbl_st.configure(text="done", foreground="blue"); self._update_ui(False); return
        elif s == "failed": self.lbl_st.configure(text=f"failed ({self.train_proc.error})", foreground="red"); self._update_ui(False); return
        elif s == "idle": self._update_ui(False); return
        self._monitor_job = self.root.after(2000, self._start_monitor)

    def _update_ui(self, r):
        self.btn_launch.configure(state="disabled" if r else "normal")
        self.btn_stop.configure(state="normal" if r else "disabled")

    def _clear_console(self): self.tc.configure(state="normal"); self.tc.delete("1.0","end"); self.tc.configure(state="disabled")
    def _append_console(self, t): self.tc.configure(state="normal"); self.tc.insert("end",t+"\n"); self.tc.see("end"); self.tc.configure(state="disabled")

    # ===== TAB 3: Visualization =====
    def _build_viz_tab(self):
        top = ttk.Frame(self.tab_viz); top.pack(fill="x", padx=8, pady=4)
        ttk.Label(top, text="Runs:", font=("",10)).pack(side="left", padx=4)
        self.viz_lb = tk.Listbox(top, selectmode="extended", height=6, width=50)
        self.viz_lb.pack(side="left", fill="y", expand=True, padx=4)
        ttk.Button(top, text="Refresh", command=self._viz_refresh).pack(side="left", padx=4)
        mf = ttk.Frame(self.tab_viz); mf.pack(fill="x", padx=8, pady=2)
        self.viz_m = tk.StringVar(value="mean_success_rate")
        ttk.Label(mf, text="Metric:").pack(side="left")
        ttk.Combobox(mf, textvariable=self.viz_m, values=["mean_success_rate","mean_reward","critic_loss","actor_loss","entropy","mean_served_keys","mean_ratio","kl"], state="readonly", width=20).pack(side="left", padx=4)
        ttk.Button(mf, text="Plot", command=self._viz_plot).pack(side="left", padx=4)
        self.viz_fig = Figure(figsize=(10,5), dpi=80)
        self.viz_ax = self.viz_fig.add_subplot(111); self.viz_ax.grid(True, alpha=0.3); self.viz_fig.tight_layout()
        self.viz_canv = FigureCanvasTkAgg(self.viz_fig, master=self.tab_viz)
        self.viz_canv.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=4)
        self._viz_refresh()

    def _viz_refresh(self):
        self.viz_lb.delete(0,"end")
        for c in list_checkpoints():
            if c["has_metrics"]: self.viz_lb.insert("end", c["name"])

    def _viz_plot(self):
        sel = [self.viz_lb.get(i) for i in self.viz_lb.curselection()]
        if not sel: messagebox.showinfo("", "select runs"); return
        field = self.viz_m.get(); self.viz_ax.clear()
        for name in sel:
            for c in list_checkpoints():
                if c["name"]==name and c["metrics_path"]:
                    try:
                        lines = [json.loads(l) for l in open(c["metrics_path"], encoding="utf-8") if l.strip()]
                        recs = [l for l in lines if "update" in l and field in l]
                        if recs: self.viz_ax.plot([r["update"] for r in recs], [r[field] for r in recs], label=name, lw=1.5)
                    except: pass
        self.viz_ax.set_xlabel("Update"); self.viz_ax.set_ylabel(field); self.viz_ax.legend(); self.viz_ax.grid(True, alpha=0.3)
        self.viz_fig.tight_layout(); self.viz_canv.draw()

    # ===== TAB 4: Evaluation =====
    def _build_eval_tab(self):
        # Top: basic config
        cf = ttk.LabelFrame(self.tab_eval, text="Eval Config", padding=8); cf.pack(fill="x", padx=8, pady=4)
        r=0
        ttk.Label(cf, text="全局验证基准:", font=("", 9, "bold")).grid(row=r, column=0, columnspan=4, sticky="w", padx=4, pady=(4,0)); r+=1
        self.e_global_info = ttk.Label(cf, text="", foreground="gray")
        self.e_global_info.grid(row=r, column=0, columnspan=4, sticky="w", padx=4); r+=1
        ttk.Separator(cf, orient="horizontal").grid(row=r, column=0, columnspan=4, sticky="ew", pady=4); r+=1

        ttk.Label(cf, text="额外种子(追加):").grid(row=r, column=0, sticky="w", padx=4, pady=2)
        self.e_extra_seeds = tk.StringVar(value="")
        ttk.Entry(cf, textvariable=self.e_extra_seeds, width=20).grid(row=r, column=1, sticky="w", padx=4, pady=2)
        self.e_total_label = ttk.Label(cf, text="总种子: [7]", foreground="blue")
        self.e_total_label.grid(row=r, column=2, columnspan=2, sticky="w", padx=4); r+=1

        ttk.Label(cf, text="episodes/种子:").grid(row=r, column=0, sticky="w", padx=4, pady=2)
        self.e_ep = tk.StringVar(value="1")
        ttk.Entry(cf, textvariable=self.e_ep, width=10).grid(row=r, column=1, sticky="w", padx=4, pady=2)
        ttk.Label(cf, text="=1匹配训练验证", foreground="gray").grid(row=r, column=2, columnspan=2, sticky="w", padx=4); r+=1

        ttk.Label(cf, text="输出目录:").grid(row=r, column=0, sticky="w", padx=4, pady=2)
        self.e_ou = tk.StringVar(value="ui_eval_result")
        ttk.Entry(cf, textvariable=self.e_ou, width=30).grid(row=r, column=1, sticky="w", padx=4, pady=2); r+=1

        self.e_extra_seeds.trace_add("write", self._update_eval_total)

        # Algorithm list: baselines + RL model
        bl_frame = ttk.LabelFrame(self.tab_eval, text="参与对比的算法", padding=8)
        bl_frame.pack(fill="x", padx=8, pady=4)
        cv = tk.Canvas(bl_frame, height=250)
        sb = ttk.Scrollbar(bl_frame, orient="vertical", command=cv.yview)
        sf = ttk.Frame(cv)
        sf.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))
        cv.create_window((0, 0), window=sf, anchor="nw"); cv.configure(yscrollcommand=sb.set)
        cv.pack(side="left", fill="both", expand=True); sb.pack(side="right", fill="y")
        cv.bind("<MouseWheel>", lambda e: cv.yview_scroll(int(-1*(e.delta/120)), "units"))

        self.e_algo_vars = {}  # {name: {param: var}}
        bl_cfg = load_baselines_config()
        algo_row = 0

        # Baseline algorithms. Params wrap onto a second row when many:
        # checkbox in col 0, params in cols 1..N; if a policy has > PARAMS_PER_ROW
        # parameters they continue on the next row, keeping the UI compact.
        PARAMS_PER_ROW = 3  # 3 params = 6 columns (label + entry each)

        def _param_cols(param_count: int) -> int:
            return 1 + param_count * 2

        for bl in get_baselines():
            bl_yaml = bl_cfg.get(bl, {})
            enabled_val = bl_yaml.get("_enabled", 1 if bl in ("greedy_demand","greedy_relay_diffusion_v3") else 0)
            enabled = tk.IntVar(value=enabled_val)
            ttk.Checkbutton(sf, text=bl, variable=enabled, onvalue=1, offvalue=0).grid(row=algo_row, column=0, sticky="w", padx=4)
            self.e_algo_vars[bl] = {"_enabled": enabled, "_type": "baseline"}
            params = [(k, v) for k, v in bl_yaml.items() if k not in ("enabled", "_enabled", "_type")]
            total_cols = _param_cols(len(params))
            col = 1
            for i, (k, v) in enumerate(params):
                # wrap to next row every PARAMS_PER_ROW params
                if i > 0 and i % PARAMS_PER_ROW == 0:
                    algo_row += 1
                    col = 1
                var = tk.StringVar(value=str(v))
                ttk.Label(sf, text=f"{k}:").grid(row=algo_row, column=col, padx=2, sticky="w")
                ttk.Entry(sf, textvariable=var, width=8).grid(row=algo_row, column=col+1, padx=2)
                self.e_algo_vars[bl][k] = var
                col += 2
            # keep same column count across rows for alignment
            algo_row += 1

        # RL model line (separator + checkbox + file browser)
        ttk.Separator(sf, orient="horizontal").grid(row=algo_row, column=0, columnspan=20, sticky="ew", pady=4); algo_row += 1
        self.e_rl_enabled = tk.IntVar(value=0)
        ttk.Checkbutton(sf, text="rl_model", variable=self.e_rl_enabled, onvalue=1, offvalue=0).grid(row=algo_row, column=0, sticky="w", padx=4)
        self.e_algo_vars["rl_model"] = {"_enabled": self.e_rl_enabled, "_type": "rl"}
        self.e_rl_ckpt = tk.StringVar(value="")
        ttk.Label(sf, text="checkpoint:").grid(row=algo_row, column=1, sticky="w", padx=2)
        ttk.Label(sf, textvariable=self.e_rl_ckpt, foreground="gray", width=25).grid(row=algo_row, column=2, columnspan=2, sticky="w", padx=2)
        ttk.Button(sf, text="浏览", command=self._browse_rl_ckpt, width=6).grid(row=algo_row, column=4, padx=2)
        algo_row += 1

        # Buttons
        bf = ttk.Frame(self.tab_eval); bf.pack(fill="x", padx=8, pady=4)
        self.e_st = ttk.Label(bf, text="", foreground="green")
        self.e_st.pack(side="left", padx=4)
        ttk.Button(bf, text="保存配置", command=self._save_baselines, width=15).pack(side="left", padx=4)
        self.btn_eval = ttk.Button(bf, text="运行验证", command=self._run_eval, width=15)
        self.btn_eval.pack(side="left", padx=4)

        # Console output (like training tab)
        console_frame = ttk.LabelFrame(self.tab_eval, text="控制台输出", padding=4)
        console_frame.pack(fill="x", padx=8, pady=2)
        self.e_console = tk.Text(console_frame, height=6, wrap="word", state="disabled",
                                  bg="#1e1e1e", fg="#d4d4d4", font=("Consolas", 9))
        self.e_console.pack(fill="both", expand=True)

        # Results summary
        rf = ttk.LabelFrame(self.tab_eval, text="验证结果", padding=8)
        rf.pack(fill="both", expand=True, padx=8, pady=4)
        self.e_tx = tk.Text(rf, height=8, wrap="word", state="disabled", font=("Consolas",10))
        self.e_tx.pack(fill="both", expand=True)

    def _browse_rl_ckpt(self):
        p = filedialog.askopenfilename(title="选择RL模型权重", initialdir=str(ROOT/"outputs"), filetypes=[("pt","*.pt"),("*","*.*")])
        if p: self.e_rl_ckpt.set(p)

    def _get_global_seeds(self) -> list[int]:
        """Return the base validation seeds from global config."""
        try:
            p = ROOT / "configs" / "global.yaml"
            c = yaml.safe_load(open(p, encoding="utf-8")) or {}
            val = c.get("global", {}).get("validation", {})
            return [int(s) for s in val.get("request_seeds", [7])]
        except: return [7]

    def _update_eval_total(self, *args):
        """Update the total seeds label."""
        base = self._get_global_seeds()
        extra = [int(s.strip()) for s in self.e_extra_seeds.get().split(",") if s.strip()]
        total = base + extra
        self.e_total_label.configure(text=f"总种子: {total}")

    def _save_baselines(self):
        try:
            cfg = {}
            for bl in get_baselines():
                if bl == "milp":
                    continue  # milp = offline ideal upper bound, no env-replay params
                d = {k: v.get() for k, v in self.e_algo_vars[bl].items() if k not in ("_enabled", "_type", "enabled")}
                # Save checkbox state (display filters it out)
                d["_enabled"] = self.e_algo_vars[bl]["_enabled"].get()
                cfg[bl] = d
            save_baselines_config(cfg)
            self.e_st.configure(text="saved", foreground="green")
            self.root.after(2000, lambda: self.e_st.configure(text=""))
        except Exception as e: self.e_st.configure(text=f"err: {e}", foreground="red")

    def _run_eval(self):
        self._save_baselines()
        # Calculate total seeds: base + extra
        base_seeds = self._get_global_seeds()
        extra = [int(s.strip()) for s in self.e_extra_seeds.get().split(",") if s.strip()]
        all_seeds = base_seeds + extra
        seeds_str = ",".join(str(s) for s in all_seeds)

        # Check which baselines are enabled
        enabled_policies = [bl for bl in get_baselines() if self.e_algo_vars[bl]["_enabled"].get() == 1]
        # MILP is the offline ideal upper bound: run it via
        # compute_milp_upper_bound.py (not the env-replay milp in run_baselines).
        milp_enabled = "milp" in enabled_policies
        if milp_enabled:
            enabled_policies = [p for p in enabled_policies if p != "milp"]

        # Check RL model
        ckpt = self.e_rl_ckpt.get().strip() or None
        if ckpt and self.e_rl_enabled.get() == 1:
            enabled_policies.append("rl_model")

        self.e_console.configure(state="normal"); self.e_console.delete("1.0","end"); self.e_console.configure(state="disabled")
        self.e_tx.configure(state="normal"); self.e_tx.delete("1.0","end")
        self.e_tx.insert("end",f"全局种子: {base_seeds}\n额外种子: {extra}\n总种子: {all_seeds}\nepisodes/种子: {self.e_ep.get()}\n")
        self.e_tx.insert("end",f"参与的算法: {enabled_policies}\nRL模型: {ckpt or '未选择'}\n\n")
        self.e_tx.insert("end","running...\n"); self.e_tx.configure(state="disabled")
        self.btn_eval.configure(state="disabled")
        def _run():
            import subprocess, json as _json
            try:
                od = ROOT/"outputs"/"eval"/self.e_ou.get().strip(); od.mkdir(parents=True, exist_ok=True)
                cmd = [sys.executable, "scripts/run_baselines.py",
                    f"--episodes={self.e_ep.get()}", f"--seeds={seeds_str}",
                    f"--out=outputs/eval/{self.e_ou.get().strip()}"]
                if enabled_policies:
                    cmd.append(f"--policies={','.join(enabled_policies)}")
                if ckpt and self.e_rl_enabled.get() == 1:
                    cmd.append(f"--rl-checkpoint={ckpt}")
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=str(ROOT), text=True, bufsize=1,
                    env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"})
                self._eval_proc = proc
                # Save PID
                pid_path = od / ".eval_pid"
                try: pid_path.write_text(str(proc.pid), encoding="utf-8")
                except: pass
                for line in iter(proc.stdout.readline, ""):
                    self.root.after(0, self._append_eval_line, line.rstrip())
                proc.wait()

                # If MILP upper bound requested, run the offline ideal solver and
                # merge its result into the baselines summary as policy "milp_ub".
                if milp_enabled:
                    self.root.after(0, self._append_eval_line, "running MILP offline upper bound (ideal flow)...")
                    ub_out = od / "milp_ub"
                    cmd2 = [sys.executable, "scripts/compute_milp_upper_bound.py",
                        "--window-steps", "360", "--time-limit", "120",
                        "--max-requests", "512", "--max-paths", "256", "--max-hops", "10",
                        f"--out=outputs/eval/{self.e_ou.get().strip()}/milp_ub"]
                    proc2 = subprocess.Popen(cmd2, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=str(ROOT), text=True, bufsize=1,
                        env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"})
                    for line in iter(proc2.stdout.readline, ""):
                        self.root.after(0, self._append_eval_line, line.rstrip())
                    proc2.wait()
                    ub_summary = ub_out / "summary.json"
                    if ub_summary.is_file():
                        ub = _json.loads(ub_summary.read_text(encoding="utf-8"))
                        self._merge_milp_ub_into_summary(od / "summary.json", ub)

                sp = od/"summary.json"
                if sp.is_file():
                    self.root.after(0, self._show_eval_summary, json.loads(sp.read_text(encoding="utf-8")))
            except Exception as e: self.root.after(0, self._append_eval_line, f"err: {e}")
            self.root.after(0, lambda: self.btn_eval.configure(state="normal"))
        threading.Thread(target=_run, daemon=True).start()

    def _merge_milp_ub_into_summary(self, summary_path: Path, ub: dict) -> None:
        """Merge the MILP ideal upper bound into the baselines summary, aligned
        with the evaluator's aggregate format. It OVERWRITES any stale
        env-replay 'milp' entry so the summary has a single MILP row (the
        offline ideal upper bound)."""
        if not summary_path.is_file():
            return
        try:
            import json as _json
            data = _json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            data = {"meta": {}, "policies": {}}
        policies = data.setdefault("policies", {})
        served = float(ub.get("total_served", 0.0))
        arrived = float(ub.get("total_arrived", 1e-9))
        sr = float(ub.get("success_rate_upper_bound", 0.0))
        # Same aggregate field set as evaluator.aggregate_episodes, so the row
        # is directly comparable with the other baselines.
        policies["milp"] = {
            "n_episodes": 1,
            "total_reward_mean": 0.0,
            "total_reward_std": 0.0,
            "arrived_keys_mean": arrived,
            "arrived_keys_std": 0.0,
            "served_keys_mean": served,
            "served_keys_std": 0.0,
            "failed_keys_mean": max(0.0, arrived - served),
            "failed_keys_std": 0.0,
            "success_rate_mean": sr,
            "success_rate_std": 0.0,
            "conflict_count_mean": 0.0,
            "conflict_count_std": 0.0,
            "steps_mean": float(ub.get("window_steps", 360)),
            "steps_std": 0.0,
            "note": "offline ideal upper bound (no env replay)",
            "runs": [{
                "timestamp": str(ub.get("seed", 7)),
                "window_steps": ub.get("window_steps", 360),
                "episode_log": [{
                    "policy": "milp", "episode": 0, "seed": ub.get("seed", 7),
                    "steps": int(ub.get("window_steps", 360)),
                    "total_reward": 0.0,
                    "arrived_keys": arrived,
                    "served_keys": served,
                    "failed_keys": max(0.0, arrived - served),
                    "success_rate": sr,
                    "conflict_count": 0,
                }],
            }],
        }
        data.setdefault("meta", {})["has_milp_ub"] = True
        summary_path.write_text(_json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def _append_eval_line(self, t):
        # Console: dark terminal style
        self.e_console.configure(state="normal"); self.e_console.insert("end",t+"\n"); self.e_console.see("end"); self.e_console.configure(state="disabled")
        # Results: also log
        self.e_tx.configure(state="normal"); self.e_tx.insert("end",t+"\n"); self.e_tx.see("end"); self.e_tx.configure(state="disabled")

    def _show_eval_summary(self, s):
        self.e_tx.configure(state="normal")
        self.e_tx.insert("end","\n"+"="*60+"\nRESULTS\n"+"="*60+"\n")
        self.e_tx.insert("end",f"{'policy':<30} {'SR':<10} {'served':<15} {'failed':<15}\n")
        self.e_tx.insert("end","-"*70+"\n")
        for p, d in sorted(s.items()):
            self.e_tx.insert("end",f"{p:<30} {d.get('success_rate_mean',0):<10.4f} {d.get('served_keys_mean',0):<15,.0f} {d.get('failed_keys_mean',0):<15,.0f}\n")
        self.e_tx.configure(state="disabled")

    def _on_close(self):
        running = []
        if self.train_proc and self.train_proc.status == "running":
            running.append("训练")
        if getattr(self, '_eval_proc', None) and self._eval_proc.poll() is None:
            running.append("验证")
        if running:
            msg = f"{'和'.join(running)}正在运行，关闭UI后会继续在后台运行。\n\n确认关闭？"
            keep = messagebox.askyesno("进程运行中", msg, icon="warning")
            if not keep:
                return
            # Detach training
            if self.train_proc and self.train_proc.status == "running":
                self.train_proc._process = None
            # Detach eval
            if getattr(self, '_eval_proc', None):
                self._eval_proc = None
            # Save restart script
            cmd_path = ROOT / "outputs" / "restart_ui.bat"
            try:
                cmd_path.write_text('@echo off\ncall D:\\anaconda1\\envs\\pytorch\\python.exe ui_tk/app.py\npause', encoding="utf-8")
            except Exception:
                pass
        if self._monitor_job: self.root.after_cancel(self._monitor_job)
        self.root.destroy()


if __name__ == "__main__":
    QKDRLApp()
