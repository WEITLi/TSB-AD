import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import math
import networkx as nx
from matplotlib.backends.backend_agg import FigureCanvasAgg
import seaborn as sns

from statistical_analysis import form_cliques

def graph_ranks(avranks, names, p_values, cd=None, cdmethod=None, lowv=None, highv=None,
                width=5, textspace=1.5, reverse=False, filename=None, **kwargs):
    """
    绘制临界差异图，展示算法排名和统计显著性差异
    
    参数:
        avranks: 算法平均排名的数组
        names: 算法名称的数组
        p_values: 成对 p 值结果的列表
        cd: 临界差异值（如果为 None 则不绘制）
        width: 图形宽度
        textspace: 文本与图之间的间距
        reverse: 是否反转排名（默认为 False）
    """
    width = float(width)
    textspace = float(textspace)
    
    # 辅助函数：从列表中获取第 n 列
    def nth(l, n):
        n = lloc(l, n)
        return [a[n] for a in l]
    
    # 辅助函数：计算位置
    def lloc(l, n):
        if n < 0:
            return len(l[0]) + n
        else:
            return n
    
    # 辅助函数：生成范围迭代器
    def mxrange(lr):
        if not len(lr):
            yield ()
        else:
            index = lr[0]
            if isinstance(index, int):
                index = [index]
            for a in range(*index):
                for b in mxrange(lr[1:]):
                    yield tuple([a] + list(b))

    # 排名值
    sums = avranks
    nnames = names
    ssums = sums
    
    # 设置排名范围
    if lowv is None:
        lowv = min(1, int(math.floor(min(ssums))))
    if highv is None:
        highv = max(len(avranks), int(math.ceil(max(ssums))))

    cline = 0.4  # 中心线位置
    k = len(sums)  # 算法数量
    
    # 计算排名位置
    def rankpos(rank):
        if not reverse:
            a = rank - lowv
        else:
            a = highv - rank
        return textspace + scalewidth / (highv - lowv) * a
    
    # 设置图形格式
    scalewidth = width - 2 * textspace
    distanceh = 0.25
    cline += distanceh
    
    minnotsignificant = max(2 * 0.2, 0)
    height = cline + ((k + 1) / 2) * 0.2 + minnotsignificant + 1.5
    
    # 创建图形
    fig = plt.figure(figsize=(width, height))
    fig.set_facecolor('white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    
    # 辅助函数：高度和宽度标准化
    hf = 1. / height
    wf = 1. / width
    
    def hfl(l):
        return [a * hf for a in l]
    
    def wfl(l):
        return [a * wf for a in l]
    
    # 设置图形边界
    ax.plot([0, 1], [0, 1], c="w")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    
    # 辅助函数：绘制线
    def line(l, color='k', **kwargs):
        ax.plot(wfl(nth(l, 0)), hfl(nth(l, 1)), color=color, **kwargs)
    
    # 辅助函数：添加文本
    def text(x, y, s, *args, **kwargs):
        ax.text(wf * x, hf * y, s, *args, **kwargs)
    
    # 绘制中心线
    line([(textspace, cline), (width - textspace, cline)], linewidth=0.7)
    
    bigtick = 0.1
    smalltick = 0.05
    linewidth = 2.0
    linewidth_sign = 4.0
    
    # 绘制刻度线
    for a in list(np.arange(lowv, highv, 0.5)) + [highv]:
        tick = smalltick
        if a == int(a):
            tick = bigtick
        line([(rankpos(a), cline - tick / 2),
              (rankpos(a), cline)],
             linewidth=0.7)
    
    # 添加刻度标签
    for a in range(lowv, highv + 1):
        text(rankpos(a), cline - tick / 2 - 0.05, str(a),
             ha="center", va="bottom", size=16)
    
    # 过滤函数 (可用于格式化名称)
    def filter_names(name):
        return name
    
    space_between_names = 0.24
    
    # 绘制前半部分算法线条和标签
    for i in range(math.ceil(k / 2)):
        chei = cline + minnotsignificant + i * space_between_names
        line([(rankpos(ssums[i]), cline),
              (rankpos(ssums[i]), chei),
              (textspace - 0.1, chei)],
             linewidth=linewidth)
        
        text(textspace - 0.2, chei, filter_names(nnames[i]), 
             color='k', ha="right", va="center", size=16)
    
    # 绘制后半部分算法线条和标签
    for i in range(math.ceil(k / 2), k):
        chei = cline + minnotsignificant + (k - i - 1) * space_between_names
        line([(rankpos(ssums[i]), cline),
              (rankpos(ssums[i]), chei),
              (textspace + scalewidth + 0.1, chei)],
             linewidth=linewidth)
        
        text(textspace + scalewidth + 0.2, chei, filter_names(nnames[i]), 
             color='k', ha="left", va="center", size=16)
    
    # 绘制统计显著性线段 (连接无显著差异的算法)
    start = cline + 0.2
    side = -0.02
    height = 0.1
    
    # 形成算法簇并绘制连接线
    cliques = form_cliques(p_values, nnames)
    achieved_half = False
    
    for clq in cliques:
        if len(clq) == 1:
            continue
        
        min_idx = np.array(clq).min()
        max_idx = np.array(clq).max()
        
        if min_idx >= len(nnames) / 2 and achieved_half == False:
            start = cline + 0.25
            achieved_half = True
            
        line([(rankpos(ssums[min_idx]) - side, start),
              (rankpos(ssums[max_idx]) + side, start)],
             linewidth=linewidth_sign)
             
        start += height

def plot_critical_diagram(df_perf, algorithm_list, title=None, alpha=0.05, width=5, textspace=1.5):
    """
    简化的临界差异图绘制函数，整合了数据准备和绘图
    
    参数:
        df_perf: 包含性能评估结果的DataFrame
        algorithm_list: 要比较的算法列表
        title: 图表标题
        alpha: 显著性水平
    """
    from statistical_analysis import Friedman_Nemenyi
    
    # 运行 Friedman-Nemenyi 检验
    p_values, average_ranks, _ = Friedman_Nemenyi(df_perf=df_perf, alpha=alpha)
    
    if p_values is None:
        plt.figure(figsize=(width, 3))
        plt.text(0.5, 0.5, "无统计显著差异 (p >= {})".format(alpha), 
                 ha='center', va='center', fontsize=16)
        plt.axis('off')
        if title:
            plt.title(title, fontsize=16)
        return
    
    # 按排名顺序获取算法名称
    ranking = average_ranks.keys().tolist()[::-1]
    
    # 绘制临界差异图
    graph_ranks(average_ranks.values, average_ranks.keys(), p_values,
                cd=None, reverse=True, width=width, textspace=textspace)
    
    # 添加标题
    if title:
        plt.title(title, fontsize=16)
    else:
        plt.title("临界差异图 ({}={})".format(r'$\alpha$', alpha), fontsize=16)
    
    plt.tight_layout()
    
    return ranking, average_ranks

def plot_performance_boxplot(df, algorithm_list, metric_col, title=None, figsize=(7, 4)):
    """
    绘制算法性能的箱线图
    
    参数:
        df: 包含评估结果的DataFrame
        algorithm_list: 要比较的算法列表 (按排名顺序)
        metric_col: 指标列名称
        title: 图表标题
        figsize: 图形大小
    """
    plt.figure(figsize=figsize)
    sns.reset_orig()  # 重置样式
    
    # 创建箱线图
    ax = sns.boxplot(data=df[algorithm_list], showfliers=False,
                     meanprops=dict(color='k', linestyle='--'), 
                     showmeans=True, meanline=True)
    
    # 设置标签和标题
    plt.xticks(ticks=range(len(algorithm_list)), labels=algorithm_list, 
               rotation=90, fontsize=12)
    plt.ylabel(metric_col, fontsize=12)
    
    if title:
        plt.title(title, fontsize=14)
        
    plt.tight_layout()
    
    return ax 