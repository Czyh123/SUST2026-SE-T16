import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from heapq import heappush, heappop
import os
import json

# ====================== 中文显示全局锁定 ======================
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False
CHINESE_FONT = "SimHei"
# ==============================================================

# ====================== 景游园智导项目默认数据 ======================
DEFAULT_PROJECT_TASKS = [
    {"name": "景区需求调研与分析", "optimistic": 3, "most_likely": 5, "pessimistic": 7, "predecessors": []},
    {"name": "系统架构设计", "optimistic": 2, "most_likely": 4, "pessimistic": 6, "predecessors": ["景区需求调研与分析"]},
    {"name": "UI/UX原型设计", "optimistic": 3, "most_likely": 5, "pessimistic": 8, "predecessors": ["景区需求调研与分析"]},
    {"name": "多智能体协同算法开发", "optimistic": 8, "most_likely": 12, "pessimistic": 16, "predecessors": ["系统架构设计"]},
    {"name": "客流预测模型开发", "optimistic": 6, "most_likely": 10, "pessimistic": 14, "predecessors": ["系统架构设计"]},
    {"name": "资源调度引擎开发", "optimistic": 5, "most_likely": 8, "pessimistic": 11, "predecessors": ["系统架构设计"]},
    {"name": "前端可视化界面开发", "optimistic": 6, "most_likely": 9, "pessimistic": 12, "predecessors": ["UI/UX原型设计"]},
    {"name": "系统集成与联调", "optimistic": 4, "most_likely": 6, "pessimistic": 9, "predecessors": ["多智能体协同算法开发", "客流预测模型开发", "资源调度引擎开发", "前端可视化界面开发"]},
    {"name": "系统测试与Bug修复", "optimistic": 5, "most_likely": 7, "pessimistic": 10, "predecessors": ["系统集成与联调"]},
    {"name": "疏导员培训教材编写", "optimistic": 3, "most_likely": 5, "pessimistic": 7, "predecessors": ["系统测试与Bug修复"]},
    {"name": "系统部署与试运行", "optimistic": 2, "most_likely": 3, "pessimistic": 5, "predecessors": ["系统测试与Bug修复", "疏导员培训教材编写"]}
]
# ========================================================================

class PERTProject:
    def __init__(self):
        self.tasks = {}
        self.expected_durations = {}
        self.es = {}  # 最早开始时间 (Earliest Start)
        self.ef = {}  # 最早结束时间 (Earliest Finish)
        self.ls = {}  # 最晚开始时间 (Latest Start)
        self.lf = {}  # 最晚结束时间 (Latest Finish)
        self.float_time = {} # 总时差/浮动时间
        self.critical_path = []
        self.project_duration = 0
        self.G = nx.DiGraph()

    def add_task(self, name, optimistic, most_likely, pessimistic, predecessors=None):
        if predecessors is None: predecessors = []
        self.tasks[name] = (optimistic, most_likely, pessimistic, predecessors)
        # PERT三点估算法：期望工期 = (乐观 + 4*最可能 + 悲观) / 6
        self.expected_durations[name] = (optimistic + 4 * most_likely + pessimistic) / 6
        self.G.add_node(name, duration=self.expected_durations[name])
        for pred in predecessors: self.G.add_edge(pred, name)

    def batch_add_tasks(self, task_list):
        for task in task_list:
            self.add_task(task["name"], task["optimistic"], task["most_likely"], task["pessimistic"], task["predecessors"])
        print(f"✅ 批量添加{len(task_list)}个任务完成")

    def calculate_schedule(self):
        """
        【核心算法1】：CPM关键路径法 - 正推法与逆推法
        """
        # --- 第一步：正推法 (Forward Pass) ---
        # 按拓扑顺序遍历，计算每个任务的最早开始(ES)和最早结束(EF)
        topological_order = list(nx.topological_sort(self.G))
        for task in topological_order:
            preds = list(self.G.predecessors(task))
            # 规则：如果没有前置任务，ES=0；否则，ES=所有前置任务EF的最大值
            self.es[task] = 0 if not preds else max(self.ef[pred] for pred in preds)
            # EF = ES + 工期
            self.ef[task] = self.es[task] + self.expected_durations[task]
        
        # 项目总工期 = 最后一个任务的EF最大值
        self.project_duration = max(self.ef.values())

        # --- 第二步：逆推法 (Backward Pass) ---
        # 按拓扑逆序遍历，计算每个任务的最晚结束(LF)和最晚开始(LS)
        reverse_order = reversed(topological_order)
        for task in reverse_order:
            succs = list(self.G.successors(task))
            # 规则：如果没有后续任务，LF=项目总工期；否则，LF=所有后续任务LS的最小值
            self.lf[task] = self.project_duration if not succs else min(self.ls[succ] for succ in succs)
            # LS = LF - 工期
            self.ls[task] = self.lf[task] - self.expected_durations[task]

        # --- 第三步：计算总时差并识别关键路径 ---
        # 总时差 Float = LS - ES
        # 关键路径：总时差为0的任务串联而成
        for task in self.tasks: self.float_time[task] = self.ls[task] - self.es[task]
        self.critical_path = [task for task in self.tasks if self.float_time[task] == 0]

    def calculate_min_resources_and_schedule(self):
        """
        【核心算法2】：资源平衡 - 基于贪心策略和最小堆(Heapq)
        目标：在不延长项目工期的前提下，计算最少需要多少人
        """
        # 按最早开始时间(ES)对任务进行排序
        task_list = sorted(self.tasks.keys(), key=lambda x: self.es[x])
        
        resource_heap = []  # 最小堆：存储 (人员空闲时间点, 人员ID)
        assignment = {}     # 任务分配结果
        resource_id = 0     # 人员ID计数器

        for task in task_list:
            start = self.es[task]
            end = self.ef[task]
            assigned = False

            # 策略1：尝试找一个已经干完活的人（空闲时间 <= 当前任务开始时间）
            while resource_heap:
                # 从堆顶取出最早空闲的人
                avail_time, r_id = heappop(resource_heap)
                
                if avail_time <= start:
                    # 这个人有空，把任务分配给他
                    if r_id not in assignment: assignment[r_id] = []
                    assignment[r_id].append((task, start, end))
                    # 更新这个人的新空闲时间，并重新压入堆
                    heappush(resource_heap, (end, r_id))
                    assigned = True
                    break
                else:
                    # 这个人还没空，放回去，不找了（因为是最小堆，后面的人只会更忙）
                    heappush(resource_heap, (avail_time, r_id))
                    break
            
            # 策略2：如果找不到空闲的人，就新增一个人
            if not assigned:
                assignment[resource_id] = [(task, start, end)]
                heappush(resource_heap, (end, resource_id))
                resource_id += 1
                
        return resource_id, assignment

    def output_results(self):
        print("\n" + "="*70 + "\n📊 PERT项目计算结果\n" + "="*70)
        print("\n1. 各任务期望工期：")
        for task, dur in self.expected_durations.items():
            tag = "【关键】" if task in self.critical_path else ""
            print(f"   {task}: {dur:.2f}天 {tag}")
        print(f"\n2. 关键路径：\n   {' → '.join(self.critical_path)}")
        print(f"\n3. 项目总工期：{self.project_duration:.2f}天")
        min_people, assignment = self.calculate_min_resources_and_schedule()
        print(f"\n4. 最少人数：{min_people}人")
        return pd.DataFrame([{"人员":f"人员{r+1}", "任务":t, "开始":f"{s:.1f}", "结束":f"{e:.1f}"} 
                            for r, ts in assignment.items() for t, s, e in ts]), assignment

    def visualize(self, assignment_df, assignment):
        # --- 窗口1：PERT网络图 (缩小适配版) ---
        fig1 = plt.figure("PERT网络图", figsize=(18, 10))
        ax1 = fig1.add_subplot(111)

        # ========== 手动拓扑分层布局 (零重叠) ==========
        topological_order = list(nx.topological_sort(self.G))
        task_level = {}
        for task in topological_order:
            preds = list(self.G.predecessors(task))
            if not preds:
                task_level[task] = 0
            else:
                task_level[task] = max(task_level[pred] for pred in preds) + 1

        level_tasks = {}
        for task, level in task_level.items():
            if level not in level_tasks:
                level_tasks[level] = []
            level_tasks[level].append(task)

        # 配套缩小间距，适配小画布
        pos = {}
        x_spacing = 4.5  # 从6缩小到4.5
        y_spacing = 3    # 从4缩小到3
        max_level = max(level_tasks.keys())

        for level, tasks in level_tasks.items():
            x = level * x_spacing
            task_count = len(tasks)
            y_start = (task_count - 1) * y_spacing / 2
            for idx, task in enumerate(tasks):
                y = y_start - idx * y_spacing
                pos[task] = (x, y)

        # ========== 绘图样式 (适配小画布) ==========
        node_colors = ["#ff4444" if t in self.critical_path else "#81d4fa" for t in self.G.nodes]
        # 关键路径边识别
        critical_edges = []
        for i in range(len(self.critical_path)-1):
            u = self.critical_path[i]
            v = self.critical_path[i+1]
            if self.G.has_edge(u, v):
                critical_edges.append((u, v))
        non_critical_edges = [e for e in self.G.edges if e not in critical_edges]

        # 1. 先画非关键路径箭头
        nx.draw_networkx_edges(
            self.G, pos, ax=ax1, edgelist=non_critical_edges,
            edge_color="#444444",
            width=2,
            arrowstyle='->',
            arrowsize=20
        )
        # 2. 再画关键路径箭头
        nx.draw_networkx_edges(
            self.G, pos, ax=ax1, edgelist=critical_edges,
            edge_color="#ff2222",
            width=3.5,
            arrowstyle='->',
            arrowsize=25
        )
        # 3. 画节点 (缩小尺寸适配画布)
        nx.draw_networkx_nodes(
            self.G, pos, ax=ax1,
            node_size=2800, node_color=node_colors,
            edgecolors="#111", linewidths=2
        )
        nx.draw_networkx_labels(
            self.G, pos, ax=ax1,
            font_size=9, font_weight="bold", font_family=CHINESE_FONT
        )
        # 边的工期标签
        edge_labels = {(u, v): f"{self.expected_durations[v]:.1f}天" for u, v in self.G.edges}
        nx.draw_networkx_edge_labels(
            self.G, pos, edge_labels=edge_labels, ax=ax1,
            font_size=9, font_weight="bold", font_family=CHINESE_FONT,
            label_pos=0.3, rotate=False
        )

        ax1.set_title("PERT任务网络图（ 红色为关键路径）", 
                      fontsize=18, fontfamily=CHINESE_FONT, pad=20)
        ax1.margins(0.05)
        ax1.axis("off")
        plt.tight_layout()

        # --- 窗口2：甘特图 (缩小适配版) ---
        # 核心修改：画布从22*12缩小到18*9
        fig2 = plt.figure("甘特图", figsize=(18, 9))
        ax2 = fig2.add_subplot(111)
        fig2.subplots_adjust(left=0.08, right=0.95, top=0.9, bottom=0.1)
        
        person_list = sorted(assignment.keys(), reverse=True)
        person_names = [f"人员{i+1}" for i in person_list]
        colors = plt.cm.tab10(range(len(person_list)))
        color_map = {f"人员{i+1}": colors[i] for i in person_list}

        y_ticks = range(len(person_names))
        ax2.set_yticks(y_ticks)
        ax2.set_yticklabels(person_names, fontsize=12, fontfamily=CHINESE_FONT)
        
        for y_idx, r_id in enumerate(reversed(person_list)):
            person_name = f"人员{r_id+1}"
            for task, start, end in assignment[r_id]:
                duration = end - start
                ax2.barh(y_idx, duration, left=start, height=0.65, color=color_map[person_name], edgecolor="black", alpha=0.85)
                
                # 智能文字适配
                if duration > 10: font_size = 10
                elif duration > 6: font_size = 9
                elif duration > 3: font_size = 8
                else: font_size = 7
                
                if duration >= 3:
                    ax2.text((start + end)/2, y_idx, task, ha='center', va='center', 
                            color='white', fontweight='bold', fontsize=font_size, fontfamily=CHINESE_FONT)
                else:
                    ax2.text(end + 0.3, y_idx, task, ha='left', va='center', 
                            color='black', fontweight='bold', fontsize=9, fontfamily=CHINESE_FONT)

        ax2.set_xlabel("项目时间（天）", fontsize=13, fontfamily=CHINESE_FONT, labelpad=15)
        ax2.set_title("人员任务分配甘特图", fontsize=18, fontfamily=CHINESE_FONT, pad=20)
        ax2.grid(axis='x', linestyle='--', alpha=0.7)
        ax2.set_xlim(-2, self.project_duration + 10)
        ax2.set_ylim(-0.8, len(person_names)-0.2)
        
        plt.tight_layout()
        plt.show()

# ------------------------------
# 主程序
# ------------------------------
if __name__ == "__main__":
    print("🚀 景游园智导 PERT系统")
    print("="*50 + "\n1. 一键加载默认数据\n2. 手动输入\n3. JSON导入\n" + "="*50)
    project = PERTProject()
    choice = input("请选择：").strip()
    
    if choice == "1":
        project.batch_add_tasks(DEFAULT_PROJECT_TASKS)
    elif choice == "2":
        while True:
            name = input("\n任务名（q结束）：").strip()
            if name.lower() == 'q': break
            try:
                o = float(input("乐观时间："))
                m = float(input("最可能时间："))
                p = float(input("悲观时间："))
            except ValueError:
                print("请输数字！")
                continue
            preds = input("前置任务（逗号分隔）：").strip()
            project.add_task(name, o, m, p, [x.strip() for x in preds.split(',')] if preds else [])
            print(f"✅ {name} 已添加")
    elif choice == "3":
        fp = input("JSON路径：").strip()
        if not os.path.exists(fp):
            print("❌ 文件不存在")
            exit()
        df = pd.read_json(fp)
        for _, row in df.iterrows():
            preds = row["predecessors"]
            if isinstance(preds, str):
                preds = [p.strip() for p in preds.split(",") if p.strip()]
            elif not isinstance(preds, list):
                preds = []
            project.add_task(str(row["name"]), float(row["optimistic"]), float(row["most_likely"]), float(row["pessimistic"]), preds)
    else:
        print("❌ 无效选项")
        exit()

    if not project.tasks:
        print("❌ 无任务")
        exit()

    print("\n📈 计算中...")
    project.calculate_schedule()
    assignment_df, assignment = project.output_results()
    
    if input("\n显示图表？(y/n)：").lower() == 'y':
        project.visualize(assignment_df, assignment)