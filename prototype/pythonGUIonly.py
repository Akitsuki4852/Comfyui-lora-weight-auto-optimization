import os
import json
import time
from pathlib import Path

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

SCRIPT_ROOT = Path(__file__).resolve().parent
BATCHES_DIR = SCRIPT_ROOT.parent / "custom_nodes" / "lora-gradient-node" / "batches"
GUI_STATE_FILE = BATCHES_DIR / "gui_state.json"

# ==========================================
# 1. 扫描 safetensors 并建立外置字典
# ==========================================
def build_lora_dict(folder="."):
    """扫描文件夹内所有 .safetensors 文件，返回 {name: path} 并保存为 JSON"""
    lora_dict = {}
    for f in os.listdir(folder):
        if f.endswith(".safetensors"):
            name = os.path.splitext(f)[0]
            lora_dict[name] = os.path.abspath(os.path.join(folder, f))
    # 保存外置字典
    with open("lora_dict.json", "w", encoding="utf-8") as jf:
        json.dump(lora_dict, jf, indent=2, ensure_ascii=False)
    return lora_dict


def _ensure_batches_dir():
    BATCHES_DIR.mkdir(parents=True, exist_ok=True)


def _save_gui_state(state):
    _ensure_batches_dir()
    try:
        with GUI_STATE_FILE.open("w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def _load_gui_state():
    if not GUI_STATE_FILE.exists():
        return None
    try:
        with GUI_STATE_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

# ==========================================
# 2. CMA-ES 优化器（仅利用排序信息）
# ==========================================
class CMAESOptimizer:
    def __init__(self, dimension, bounds, batch_size=3, sigma0=0.2):
        self.N = dimension                 # 维度（LoRA 数量）
        self.bounds = np.array(bounds)     # [[min, max], ...]
        self.lam = batch_size              # 种群大小 λ
        self.mu = max(1, self.lam // 2)    # 父代数量 μ
        self.sigma0 = sigma0

        # 权重 w_i = log(μ+1/2) - log(i), i=1..μ, 归一化
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mu_eff = 1.0 / np.sum(self.weights ** 2)   # μ 的有效值

        # 学习率参数
        self.c_s = (self.mu_eff + 2) / (self.N + self.mu_eff + 5)
        self.d_s = 1.0 + 2.0 * max(0, np.sqrt((self.mu_eff - 1) / (self.N + 1)) - 1) + self.c_s
        self.c_c = (4 + self.mu_eff / self.N) / (self.N + 4 + 2 * self.mu_eff / self.N)
        self.c_1 = 2.0 / ((self.N + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(1 - self.c_1, 2 * (self.mu_eff - 2 + 1 / self.mu_eff) /
                        ((self.N + 2) ** 2 + self.mu_eff))
        # 期望的 ||N(0,I)||
        self.chiN = np.sqrt(self.N) * (1 - 1.0/(4*self.N) + 1.0/(21*self.N**2))

        self.reset()

    def reset(self):
        """随机初始化均值（在边界内偏中间区域）并重置演化路径"""
        self.mean = np.array([np.random.uniform(low + 0.25*(high-low),
                                                low + 0.75*(high-low))
                              for low, high in self.bounds])
        self.C = np.eye(self.N)          # 协方差矩阵
        self.sigma = self.sigma0
        self.p_s = np.zeros(self.N)      # 进化路径（步长控制）
        self.p_c = np.zeros(self.N)      # 进化路径（协方差控制）
        self.gen = 0                     # 代数

    def _clip(self, x):
        """将样本裁剪到边界内"""
        return np.clip(x, self.bounds[:, 0], self.bounds[:, 1])

    def _sample_one(self):
        """从当前分布采样一个点（拒绝采样保证在边界内）"""
        for _ in range(100):
            z = np.random.randn(self.N)
            y = self.C @ z                # C 是对称的，这里直接用矩阵乘法近似 sqrt(C)
            x = self.mean + self.sigma * y
            if np.all((x >= self.bounds[:, 0]) & (x <= self.bounds[:, 1])):
                return x, z, y
        # 如果一直超界，直接裁剪（罕见情况）
        z = np.random.randn(self.N)
        y = self.C @ z
        x = self._clip(self.mean + self.sigma * y)
        return x, z, y

    def sample_batch(self):
        """生成一批候选解"""
        batch = []
        self.current_z = []
        self.current_y = []
        for _ in range(self.lam):
            x, z, y = self._sample_one()
            batch.append(x)
            self.current_z.append(z)
            self.current_y.append(y)
        return batch

    def update(self, ranking):
        """
        根据排序结果更新分布。
        ranking: list，长度 = λ，ranking[i] 是第 i 个候选的排名（1 最好，λ 最差）
        """
        # 将排名转化为从最好到最差的索引顺序
        order = np.argsort(ranking)          # order[0] 是最好候选的索引
        # 选取前 μ 个
        selected_idx = order[:self.mu]
        weights = self.weights

        # 加权平均 z 和 y
        z_w = np.sum([weights[i] * self.current_z[idx] for i, idx in enumerate(selected_idx)], axis=0)
        y_w = np.sum([weights[i] * self.current_y[idx] for i, idx in enumerate(selected_idx)], axis=0)

        # 新均值 = 所选个体的加权和
        new_mean = np.sum([weights[i] * self._clip(self.mean + self.sigma * self.current_y[idx])
                           for i, idx in enumerate(selected_idx)], axis=0)
        self.mean = self._clip(new_mean)

        # 更新步长 sigma
        self.p_s = (1 - self.c_s) * self.p_s + np.sqrt(self.c_s * (2 - self.c_s) * self.mu_eff) * z_w
        sigma_ratio = np.linalg.norm(self.p_s) / self.chiN
        self.sigma *= np.exp(self.c_s / self.d_s * (sigma_ratio - 1))

        # 更新协方差矩阵
        h_s = 1 if (np.linalg.norm(self.p_s) / np.sqrt(1 - (1 - self.c_s)**(2*(self.gen+1)))
                    < (1.4 + 2/(self.N+1)) * self.chiN) else 0
        delta_h_s = (1 - h_s) * self.c_1 * self.c_c * (2 - self.c_c)  # 微小修正项

        self.p_c = (1 - self.c_c) * self.p_c + h_s * np.sqrt(self.c_c * (2 - self.c_c) * self.mu_eff) * y_w

        rank_mu_update = np.zeros((self.N, self.N))
        for i, idx in enumerate(selected_idx):
            rank_mu_update += weights[i] * np.outer(self.current_y[idx], self.current_y[idx])

        self.C = ((1 - self.c_1 - self.c_mu) * self.C
                  + self.c_1 * (np.outer(self.p_c, self.p_c) + delta_h_s * self.C)
                  + self.c_mu * rank_mu_update)

        # 保证 C 对称正定（简单对称化）
        self.C = (self.C + self.C.T) / 2

        self.gen += 1

# ==========================================
# 3. 图形界面
# ==========================================
class LoraOptimizerGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("LoRA 权重排序优化器 (CMA-ES)")
        self.geometry("800x600")

        # 加载 LoRA 字典
        self.lora_dict = build_lora_dict(".")
        if not self.lora_dict:
            messagebox.showwarning("警告", "当前文件夹未找到 .safetensors 文件")
        self.lora_names = list(self.lora_dict.keys())

        # 内部状态
        self.rows = []                    # 每一行的控件字典
        self.optimizer = None
        self.candidates = []              # 当前批次的候选权重
        self.current_step = 0             # 当前批内索引 (0 ~ batch_size-1)
        self.current_gen = 0

        # 构建界面
        self._build_ui()

    def _build_ui(self):
        # 顶部控制栏
        ctrl_frame = ttk.Frame(self, padding=5)
        ctrl_frame.pack(fill=tk.X)

        ttk.Button(ctrl_frame, text="重置（随机初始）", command=self.reset).pack(side=tk.LEFT, padx=5)

        ttk.Label(ctrl_frame, text="Batch 大小:").pack(side=tk.LEFT, padx=5)
        self.batch_var = tk.IntVar(value=3)
        batch_entry = ttk.Spinbox(ctrl_frame, from_=1, to=10, textvariable=self.batch_var, width=5)
        batch_entry.pack(side=tk.LEFT)
        batch_entry.bind("<FocusOut>", lambda e: self._on_batch_change())

        ttk.Label(ctrl_frame, text="Iter 名称:").pack(side=tk.LEFT, padx=5)
        self.iter_var = tk.StringVar(value="iter1")
        ttk.Entry(ctrl_frame, textvariable=self.iter_var, width=12).pack(side=tk.LEFT)

        self.round_label = ttk.Label(ctrl_frame, text="当前轮次: -")
        self.round_label.pack(side=tk.LEFT, padx=20)

        # LoRA 表格区域
        table_frame = ttk.LabelFrame(self, text="LoRA 配置", padding=5)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 表头
        header = ttk.Frame(table_frame)
        header.pack(fill=tk.X, pady=2)
        ttk.Label(header, text="LoRA 名称", width=25).pack(side=tk.LEFT, padx=2)
        ttk.Label(header, text="最小", width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(header, text="最大", width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(header, text="本轮建议权重", width=12).pack(side=tk.LEFT, padx=2)
        ttk.Label(header, text="操作", width=6).pack(side=tk.LEFT, padx=2)

        # 滚动区域（用于多行）
        canvas = tk.Canvas(table_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=canvas.yview)
        self.rows_frame = ttk.Frame(canvas)
        self.rows_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.rows_frame, anchor=tk.NW)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 添加 LoRA 按钮
        ttk.Button(table_frame, text="添加 LoRA", command=self._add_row).pack(pady=5)

        # 底部评分与提交
        bottom_frame = ttk.Frame(self, padding=5)
        bottom_frame.pack(fill=tk.X)

        ttk.Label(bottom_frame, text="评分输入（1最好，逗号分隔，如 1,3,2）：").pack(side=tk.LEFT)
        self.score_var = tk.StringVar()
        self.score_entry = ttk.Entry(bottom_frame, textvariable=self.score_var, width=30)
        self.score_entry.pack(side=tk.LEFT, padx=5)

        self.resume_btn = ttk.Button(bottom_frame, text="Resume Workflow", command=self._resume_workflow)
        self.resume_btn.pack(side=tk.RIGHT, padx=10)

        self.next_btn = ttk.Button(bottom_frame, text="获取下一批权重", command=self._next_batch)
        self.next_btn.pack(side=tk.RIGHT, padx=10)

        # 初始化一条空行
        self._add_row()

    def _add_row(self):
        """在表格中添加一行"""
        row_frame = ttk.Frame(self.rows_frame)
        row_frame.pack(fill=tk.X, pady=1)

        # LoRA 选择
        lora_var = tk.StringVar()
        combo = ttk.Combobox(row_frame, textvariable=lora_var, values=self.lora_names, width=22)
        combo.pack(side=tk.LEFT, padx=2)

        # 范围输入
        min_var = tk.StringVar(value="0")
        min_entry = ttk.Entry(row_frame, textvariable=min_var, width=6)
        min_entry.pack(side=tk.LEFT, padx=2)

        max_var = tk.StringVar(value="1.0")
        max_entry = ttk.Entry(row_frame, textvariable=max_var, width=6)
        max_entry.pack(side=tk.LEFT, padx=2)

        # 建议权重显示
        weight_label = ttk.Label(row_frame, text="—", width=12, relief=tk.SUNKEN)
        weight_label.pack(side=tk.LEFT, padx=2)

        # 删除按钮
        del_btn = ttk.Button(row_frame, text="删除", command=lambda r=row_frame: self._delete_row(r))
        del_btn.pack(side=tk.LEFT, padx=2)

        self.rows.append({
            "frame": row_frame,
            "lora": lora_var,
            "min": min_var,
            "max": max_var,
            "weight_label": weight_label
        })
        _save_gui_state(self._collect_gui_state())

    def _delete_row(self, row_frame):
        """删除指定行并重置优化器"""
        for i, r in enumerate(self.rows):
            if r["frame"] == row_frame:
                row_frame.destroy()
                self.rows.pop(i)
                break
        self._reset_optimizer()
        _save_gui_state(self._collect_gui_state())

    def _get_bounds(self):
        """从界面读取所有行的范围，返回 [[min, max], ...]"""
        bounds = []
        for row in self.rows:
            try:
                low = float(row["min"].get())
                high = float(row["max"].get())
            except ValueError:
                messagebox.showerror("错误", f"范围输入无效：{row['lora'].get()}")
                return None
            if low >= high:
                messagebox.showerror("错误", f"最小必须小于最大：{row['lora'].get()}")
                return None
            bounds.append([low, high])
        return bounds

    def _reset_optimizer(self):
        """根据当前行建立新的 CMA-ES 优化器"""
        bounds = self._get_bounds()
        if bounds is None or len(bounds) == 0:
            self.optimizer = None
            self.candidates = []
            self._update_display()
            return
        N = len(bounds)
        batch_size = self.batch_var.get()
        self.optimizer = CMAESOptimizer(N, bounds, batch_size=batch_size)
        self.current_gen = 1
        self.current_step = 0
        # 采样第一批
        self.candidates = self.optimizer.sample_batch()
        self._update_display()

    def reset(self):
        """重置按钮：重新随机初始化并采样"""
        if self.optimizer:
            self.optimizer.reset()
            self.current_gen = 1
            self.current_step = 0
            self.candidates = self.optimizer.sample_batch()
            self._update_display()
        else:
            self._reset_optimizer()
        _save_gui_state(self._collect_gui_state())

    def _on_batch_change(self):
        """Batch 大小改变时重置优化器"""
        if self.optimizer and self.optimizer.lam != self.batch_var.get():
            self._reset_optimizer()
            _save_gui_state(self._collect_gui_state())

    def _update_display(self):
        """刷新 UI 上的当前轮次和建议权重"""
        if self.optimizer is None or not self.candidates:
            self.round_label.config(text="当前轮次: -")
            for row in self.rows:
                row["weight_label"].config(text="—")
            return

        step = self.current_step
        total = len(self.candidates)
        self.round_label.config(text=f"当前轮次: {self.current_gen}-{step+1} (共{total}套)")

        if step < total:
            weights = self.candidates[step]
            for i, row in enumerate(self.rows):
                if i < len(weights):
                    row["weight_label"].config(text=f"{weights[i]:.2f}")
                else:
                    row["weight_label"].config(text="—")
        else:
            for row in self.rows:
                row["weight_label"].config(text="?")

    def _next_batch(self):
        """处理 '获取下一批权重' 按钮"""
        if self.optimizer is None:
            messagebox.showinfo("提示", "请先添加 LoRA 并设置范围")
            return

        batch_size = len(self.candidates)
        # 非最后一套：直接前进
        if self.current_step + 1 < batch_size:
            self.current_step += 1
            self._update_display()
            return

        # 最后一套：需要评分
        raw = self.score_var.get().strip()
        if not raw:
            messagebox.showwarning("警告", "当前轮次为批次最后一套，请先输入评分")
            return

        # 解析评分
        try:
            parts = [int(x.strip()) for x in raw.split(",")]
        except ValueError:
            messagebox.showerror("错误", "评分格式错误，请输入逗号分隔的整数")
            return

        if len(parts) != batch_size:
            messagebox.showerror("错误", f"评分数量必须等于 batch 大小 ({batch_size})")
            return

        if set(parts) != set(range(1, batch_size+1)):
            messagebox.showerror("错误", f"评分必须是 1 到 {batch_size} 的排列")
            return

        # 执行 CMA-ES 更新
        self.optimizer.update(parts)
        # 进入下一代
        self.current_gen += 1
        self.current_step = 0
        self.candidates = self.optimizer.sample_batch()
        self.score_var.set("")  # 清空评分输入
        self._update_display()

    def _collect_gui_state(self, resume=False):
        entries = []
        for row in self.rows:
            name = row["lora"].get().strip()
            if not name:
                continue
            try:
                low = float(row["min"].get())
                high = float(row["max"].get())
            except ValueError:
                low, high = 0.0, 1.0
            entries.append({"name": name, "min": low, "max": high})

        return {
            "batch_size": int(self.batch_var.get()),
            "iter_name": self.iter_var.get().strip() or "iter1",
            "loras": entries,
            "resume_requested": bool(resume),
            "resume_timestamp": time.time(),
        }

    def _resume_workflow(self):
        state = self._collect_gui_state(resume=True)
        if not state["loras"]:
            messagebox.showwarning("警告", "请先填写至少一个 LoRA 条目，然后再按 Resume Workflow")
            return
        _save_gui_state(state)
        messagebox.showinfo("已保存", "已保存 resume 状态，ComfyUI 工作流可继续执行。")

# ==========================================
# 运行
# ==========================================
if __name__ == "__main__":
    app = LoraOptimizerGUI()
    app.mainloop()