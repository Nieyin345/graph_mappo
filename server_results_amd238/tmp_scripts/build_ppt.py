"""由 极简黑白.pptx 生成 QKD-SAGIN 组会汇报 PPT。

思路：
  1. 复用模板的封面 / 目录 / 章节过渡页 / 结尾页（克隆幻灯片 XML）
  2. 内容页统一用模板的「标题和内容」版式（自带黑白转角装饰），正文用原生形状排版
  3. 全部图形（拓扑示意、管线图、KPI 卡片、表格）用原生形状绘制，保持可编辑
"""

import copy
import os
import shutil
import zipfile

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Cm, Pt

TEMPLATE = "极简黑白.pptx"
OUTPUT = "QKD-SAGIN生产端调度_组会汇报.pptx"

# ---------- 黑白灰配色 ----------
BLACK = RGBColor(0x1A, 0x1A, 0x1A)
DARK = RGBColor(0x33, 0x33, 0x33)
GRAY = RGBColor(0x80, 0x80, 0x80)
MID = RGBColor(0xB3, 0xB3, 0xB3)
LIGHT = RGBColor(0xD9, 0xD9, 0xD9)
BG = RGBColor(0xF2, 0xF2, 0xF2)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

FONT = "微软雅黑"

# 版面基准
PAGE_W, PAGE_H = 33.87, 19.05
X0, X1 = 1.9, 31.97          # 正文左右边界
BODY_TOP = 2.55              # 正文起始 y
TITLE_X, TITLE_Y = 2.07, 0.72


# ============================================================ 基础绘制工具
def tb(slide, x, y, w, h, paras, anchor=MSO_ANCHOR.TOP, wrap=True):
    """添加文本框。paras: [{'t','sz','b','c','al','sp','ls'}...]"""
    box = slide.shapes.add_textbox(Cm(x), Cm(y), Cm(w), Cm(h))
    tf = box.text_frame
    tf.word_wrap = wrap
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, spec in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = spec.get("al", PP_ALIGN.LEFT)
        p.line_spacing = spec.get("ls", 1.3)
        if spec.get("sp"):
            p.space_before = Pt(spec["sp"])
        r = p.add_run()
        r.text = spec.get("t", "")
        r.font.size = Pt(spec.get("sz", 12))
        r.font.bold = spec.get("b", False)
        r.font.color.rgb = spec.get("c", DARK)
        r.font.name = spec.get("f", FONT)
    return box


def rect(slide, x, y, w, h, fill=None, line=None, lw=0.75, rounded=True, radius=0.1):
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE
    sh = slide.shapes.add_shape(shape_type, Cm(x), Cm(y), Cm(w), Cm(h))
    if rounded:
        try:
            sh.adjustments[0] = radius
        except Exception:
            pass
    if fill is None:
        sh.fill.background()
    else:
        sh.fill.solid()
        sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = Pt(lw)
    sh.shadow.inherit = False
    sh.text_frame.text = ""
    return sh


def hline(slide, x, y, w, color=LIGHT, lw=0.75, dash=None):
    ln = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Cm(x), Cm(y), Cm(x + w), Cm(y))
    ln.line.color.rgb = color
    ln.line.width = Pt(lw)
    if dash:
        ln.line.dash_style = dash
    return ln


def seg(slide, x1, y1, x2, y2, color=GRAY, lw=0.75, dash=None):
    ln = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Cm(x1), Cm(y1), Cm(x2), Cm(y2))
    ln.line.color.rgb = color
    ln.line.width = Pt(lw)
    if dash:
        ln.line.dash_style = dash
    return ln


def dot(slide, cx, cy, d, fill=WHITE, line=BLACK, lw=1.0):
    sh = slide.shapes.add_shape(MSO_SHAPE.OVAL, Cm(cx - d / 2), Cm(cy - d / 2), Cm(d), Cm(d))
    if fill is None:
        sh.fill.background()
    else:
        sh.fill.solid()
        sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = Pt(lw)
    sh.shadow.inherit = False
    return sh


def box(slide, x, y, w, h, text, sz=12, fill=BG, line=None, tc=DARK, b=False,
        rounded=True, radius=0.12, align=PP_ALIGN.CENTER, lw=0.75):
    sh = rect(slide, x, y, w, h, fill=fill, line=line, lw=lw, rounded=rounded, radius=radius)
    tf = sh.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Cm(0.15)
    tf.margin_top = tf.margin_bottom = 0
    for i, line_txt in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = 1.25
        r = p.add_run()
        r.text = line_txt
        r.font.size = Pt(sz)
        r.font.bold = b
        r.font.color.rgb = tc
        r.font.name = FONT
    return sh


def draw_table(slide, x, y, col_w, rows, row_h, body_sz=10.5, head_sz=10.5,
               zebra=True, align=None):
    """黑白表格：首行黑底白字，其余白底细灰框。"""
    n_col = len(col_w)
    cy = y
    for ri, row in enumerate(rows):
        cx = x
        h = row_h[ri]
        for ci in range(n_col):
            cell = row[ci] if ci < len(row) else ""
            w = col_w[ci]
            al = (align[ci] if align else PP_ALIGN.LEFT)
            if ri == 0:
                rect(slide, cx, cy, w, h, fill=BLACK, line=None, rounded=False)
                tb(slide, cx + 0.22, cy + 0.12, w - 0.44, h - 0.24,
                   [{"t": cell, "sz": head_sz, "c": WHITE, "b": True, "al": al}],
                   anchor=MSO_ANCHOR.MIDDLE)
            else:
                fill = BG if (zebra and ri % 2 == 0) else WHITE
                rect(slide, cx, cy, w, h, fill=fill, line=LIGHT, lw=0.5, rounded=False)
                tb(slide, cx + 0.22, cy + 0.12, w - 0.44, h - 0.24,
                   [{"t": cell, "sz": body_sz, "c": DARK, "al": al}],
                   anchor=MSO_ANCHOR.MIDDLE)
            cx += w
        cy += h
    return cy


def section_head(slide, text, x, y, w=30.0, sz=13):
    """小节标题：一条竖黑条 + 文字。"""
    rect(slide, x, y + 0.08, 0.14, 0.5, fill=BLACK, rounded=False)
    tb(slide, x + 0.42, y, w, 0.7, [{"t": text, "sz": sz, "b": True, "c": BLACK}])


# ============================================================ 模板准备
def load_template():
    prs = Presentation(TEMPLATE)
    layouts = {l.name: l for l in prs.slide_layouts}

    # 归档需要复用的模板页（克隆 XML，避免受后续删页影响）
    keep = {"cover": 0, "toc": 1, "sec": [2, 7, 11, 15], "end": 19}
    sources = {
        "cover": copy.deepcopy(prs.slides[0]._element),
        "toc": copy.deepcopy(prs.slides[1]._element),
        "sec": [copy.deepcopy(prs.slides[i]._element) for i in (2, 7, 11, 15)],
        "end": copy.deepcopy(prs.slides[19]._element),
    }

    # 删除全部原有幻灯片
    sldIdLst = prs.slides._sldIdLst
    for sldId in list(sldIdLst):
        prs.part.drop_rel(sldId.rId)
        sldIdLst.remove(sldId)

    # 「标题和内容」版式里残留的提示文字会渲染在每页上，删掉它
    content_layout = layouts["标题和内容"]
    for shp in list(content_layout.shapes):
        if shp.has_text_frame and "请在此输入标题" in shp.text_frame.text:
            shp._element.getparent().remove(shp._element)

    return prs, layouts["空白"], content_layout, sources


def clone(prs, blank_layout, src_element):
    """把归档的模板页内容克隆到新幻灯片上。"""
    slide = prs.slides.add_slide(blank_layout)
    for shp in list(slide.shapes):
        shp._element.getparent().remove(shp._element)
    src_spTree = src_element.find(qn("p:cSld")).find(qn("p:spTree"))
    dst_spTree = slide.shapes._spTree
    for child in list(src_spTree):
        if child.tag in (qn("p:nvGrpSpPr"), qn("p:grpSpPr")):
            continue
        dst_spTree.append(copy.deepcopy(child))
    strip_watermarks(slide)
    return slide


def strip_watermarks(slide):
    """删掉模板里 1pt 的推广水印文字框。"""
    for shp in list(slide.shapes):
        if not shp.has_text_frame:
            continue
        sizes = [r.font.size.pt for p in shp.text_frame.paragraphs for r in p.runs if r.font.size]
        if sizes and max(sizes) <= 3:
            shp._element.getparent().remove(shp._element)


def set_text(shape, text, size=None, align=None):
    """替换形状文字。

    注意：新 run 必须插在 a:endParaRPr 之前，否则违反 OOXML 元素顺序，
    PowerPoint 会整段不渲染。同时把首段的 pPr 复制给新加的段落，保证多行样式一致。
    """
    tf = shape.text_frame
    first = tf.paragraphs[0]
    proto_pPr = first._p.find(qn("a:pPr"))
    proto_r = None
    for p in tf.paragraphs:
        if p.runs:
            proto_r = copy.deepcopy(p.runs[0]._r)
            break

    for p in list(tf.paragraphs)[1:]:
        p._p.getparent().remove(p._p)

    for i, line in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        for r in list(p.runs):
            r._r.getparent().remove(r._r)
        if i > 0 and proto_pPr is not None and p._p.find(qn("a:pPr")) is None:
            p._p.insert(0, copy.deepcopy(proto_pPr))
        if proto_r is not None:
            r_el = copy.deepcopy(proto_r)
            end = p._p.find(qn("a:endParaRPr"))
            if end is not None:
                end.addprevious(r_el)
            else:
                p._p.append(r_el)
            run = p.runs[0]
            run.text = line
            if size:
                run.font.size = Pt(size)
        else:
            run = p.add_run()
            run.text = line
            run.font.name = FONT
            if size:
                run.font.size = Pt(size)
        if align is not None:
            p.alignment = align
    return shape


# ============================================================ 内容页骨架
class Deck:
    def __init__(self, prs, content_layout):
        self.prs = prs
        self.cl = content_layout
        self.n = 0

    def page(self, title, subtitle=None):
        """新建内容页：标题 + 细分隔线 + 页脚 + 页码。"""
        slide = self.prs.slides.add_slide(self.cl)
        for shp in list(slide.shapes):
            shp._element.getparent().remove(shp._element)
        self.n += 1
        tb(slide, TITLE_X, TITLE_Y, 20.0, 1.2,
           [{"t": title, "sz": 22, "b": True, "c": BLACK}])
        hline(slide, TITLE_X, 2.02, X1 - TITLE_X, LIGHT, 1.0)
        if subtitle:
            tb(slide, 21.5, TITLE_Y + 0.22, 10.4, 0.8,
               [{"t": subtitle, "sz": 11, "c": GRAY, "al": PP_ALIGN.RIGHT}])
        tb(slide, TITLE_X, 18.35, 16.0, 0.5,
           [{"t": "QKD-SAGIN 生产端密钥生成链路调度 · 组会进展汇报", "sz": 9, "c": MID}])
        tb(slide, X1 - 4.0, 18.35, 4.0, 0.5,
           [{"t": "%02d" % self.n, "sz": 9, "c": MID, "al": PP_ALIGN.RIGHT}])
        return slide


# ============================================================ 三个原生图形
def draw_pipeline(slide, x, y, w):
    """方法总览管线图：启发式专家 → BC 预热 → RL 微调。"""
    bh, gap = 2.5, 1.5
    bw = (w - gap * 3) / 4.0
    stages = [
        ("BFS 需求扩散启发式", "PG-Phased 三阶段路径调度\n（行为克隆专家）", BG, DARK),
        ("行为克隆 BC 预热", "克隆专家动作\n获得高起点初始策略", WHITE, DARK),
        ("Graph-MAPPO 强化学习", "从 BC 权重 warm-start\nPPO 微调打分策略", BLACK, WHITE),
        ("调度策略", "每时隙输出\n一个全局链路匹配", WHITE, DARK),
    ]
    cx = x
    for i, (head, body, fill, tc) in enumerate(stages):
        rect(slide, cx, y, bw, bh, fill=fill, line=(None if fill == BLACK else MID), lw=0.75)
        tb(slide, cx + 0.2, y + 0.34, bw - 0.4, 0.8,
           [{"t": head, "sz": 13, "b": True, "c": tc, "al": PP_ALIGN.CENTER}])
        tb(slide, cx + 0.2, y + 1.22, bw - 0.4, 1.1,
           [{"t": body, "sz": 10, "c": tc, "al": PP_ALIGN.CENTER, "ls": 1.35}])
        if i < 3:
            seg(slide, cx + bw + 0.25, y + bh / 2, cx + bw + gap - 0.25, y + bh / 2, GRAY, 1.25)
            a = cx + bw + gap - 0.25
            seg(slide, a - 0.32, y + bh / 2 - 0.16, a, y + bh / 2, GRAY, 1.25)
            seg(slide, a - 0.32, y + bh / 2 + 0.16, a, y + bh / 2, GRAY, 1.25)
        cx += bw + gap


def draw_topology(slide, x, y, w):
    """三层 FSO 网络拓扑示意（原生形状）。"""
    n = 6
    d = 1.15
    span = w - 3.6
    xs = [x + 3.6 + span * i / (n - 1) for i in range(n)]
    y_sat, y_hap, y_gs, y_dem = y + 1.0, y + 4.6, y + 8.2, y + 10.4

    # 层标签
    for ly, label in ((y_sat, "卫星 SAT ×30"), (y_hap, "高空平台 HAP ×30"), (y_gs, "地面站 GS ×30")):
        tb(slide, x, ly - 0.42, 3.4, 0.85,
           [{"t": label, "sz": 11, "b": True, "c": BLACK}])

    # 分层底色带
    for ly in (y_sat, y_hap, y_gs):
        rect(slide, x + 3.4, ly - 1.0, w - 3.4, 2.0, fill=BG, line=None, rounded=False)

    # 链路（先画线，节点后画以盖住端点）
    for i in range(n):
        seg(slide, xs[i], y_sat + d / 2, xs[i], y_hap - d / 2, BLACK, 0.75)          # HAP-SAT
        seg(slide, xs[i], y_hap + d / 2, xs[i], y_gs - d / 2, GRAY, 0.75)           # GS-HAP
    for i in range(n - 1):
        seg(slide, xs[i] + d / 2, y_sat, xs[i + 1] - d / 2, y_sat, BLACK, 1.0)      # SAT-SAT
    for i in (0, 2, 5):
        j = min(i + 2, n - 1)
        seg(slide, xs[i], y_gs - d / 2, xs[j], y_sat + d / 2, MID, 0.75)      # GS-SAT
    for i, j in ((0, 3), (1, 4), (2, 5), (3, 5)):
        seg(slide, xs[i], y_gs + d / 2, xs[j], y_dem, MID, 0.75, MSO_LINE.DASH)     # GS-GS 需求边
        seg(slide, xs[j], y_dem, xs[j], y_gs + d / 2, MID, 0.75, MSO_LINE.DASH)

    # 节点
    for i in range(n):
        dot(slide, xs[i], y_sat, d, WHITE, BLACK, 1.0)
        dot(slide, xs[i], y_hap, d, GRAY, GRAY, 0.75)
        dot(slide, xs[i], y_gs, d, BLACK, BLACK, 0.75)

    # 图例
    ly = y_dem + 0.9
    items = [
        ("GS-HAP 物理链路", GRAY, None),
        ("HAP-SAT 物理链路", BLACK, None),
        ("GS-SAT 物理链路", MID, None),
        ("SAT-SAT 物理链路", BLACK, None),
        ("GS-GS 需求边（逻辑边）", MID, MSO_LINE.DASH),
    ]
    cx = x + 0.3
    for label, color, dash in items:
        seg(slide, cx, ly, cx + 1.1, ly, color, 1.25, dash)
        tb(slide, cx + 1.3, ly - 0.26, 4.5, 0.55, [{"t": label, "sz": 9.5, "c": DARK}])
        cx += 5.95


def draw_kpi_cards(slide, x, y, w, cards):
    """4 张 KPI 卡片：大数字 + 标题 + 说明。"""
    gap = 0.7
    cw = (w - gap * 3) / 4.0
    ch = 6.0
    cx = x
    for value, label, note in cards:
        rect(slide, cx, y, cw, ch, fill=WHITE, line=MID, lw=0.75)
        rect(slide, cx, y, cw, 0.16, fill=BLACK, rounded=False)
        tb(slide, cx + 0.4, y + 1.0, cw - 0.8, 1.8,
           [{"t": value, "sz": 30, "b": True, "c": BLACK, "al": PP_ALIGN.CENTER}])
        tb(slide, cx + 0.4, y + 2.95, cw - 0.8, 0.8,
           [{"t": label, "sz": 11.5, "b": True, "c": DARK, "al": PP_ALIGN.CENTER}])
        tb(slide, cx + 0.45, y + 3.75, cw - 0.9, 2.0,
           [{"t": note, "sz": 9.5, "c": GRAY, "al": PP_ALIGN.CENTER, "ls": 1.35}])
        cx += cw + gap


# ============================================================ 页面构建
def build(prs, blank, content, sources):
    deck = Deck(prs, content)

    # ---------- 1 封面 ----------
    slide = clone(prs, blank, sources["cover"])
    for shp in slide.shapes:
        if not shp.has_text_frame:
            continue
        t = shp.text_frame.text.strip()
        if t == "MINIMAL STYLE":
            set_text(shp, "QKD-SAGIN 天地一体化网络", size=15, align=PP_ALIGN.CENTER)
        elif t == "极简风":
            set_text(shp, "生产端调度", size=54, align=PP_ALIGN.CENTER)
        elif "20XX" in t:
            set_text(shp, "组会进展汇报 · 2026-09-16", align=PP_ALIGN.CENTER)
    deck.n = 1

    # ---------- 2 目录 ----------
    slide = clone(prs, blank, sources["toc"])
    deck.n = 2
    titles = ["场景与任务定义", "信道建模与速率", "算法与模型设计", "结果与现存问题"]
    descs = [
        "三层 FSO 网络场景、关键定义与双端口约束下的决策任务",
        "链路速率生成链路、协议模型与可用性判定（本部分待补充）",
        "BFS 需求扩散启发式与 Graph-MAPPO 强化学习，重点在全局匹配动作",
        "各算法成功率对比，以及强化学习全局训练停滞的现状与诊断",
    ]
    tboxes, dboxes = [], []
    for shp in slide.shapes:
        if not shp.has_text_frame:
            continue
        t = shp.text_frame.text
        if "请在此输入标题" in t:
            tboxes.append(shp)
        elif "enter the relevant" in t:
            dboxes.append(shp)
    tboxes.sort(key=lambda s: (s.left, s.top))
    dboxes.sort(key=lambda s: (s.top, s.left))
    # 标题框各含两行，按位置对应；描述框按视觉位置重排
    order = [0, 2, 1, 3]
    for i, shp in enumerate(tboxes):
        set_text(shp, "\n".join(titles[i * 2:i * 2 + 2]))
    for k, shp in enumerate(dboxes):
        set_text(shp, descs[order[k] if k < len(order) else k])


# ============================================================ 内容页明细
def page_scenario(deck):
    s = deck.page("场景：QKD-SAGIN 三层 FSO 网络", "第一部分 · 场景分析与任务定义")
    tb(s, X0, BODY_TOP, X1 - X0, 1.8, [{
        "t": "天地一体化量子密钥分发网络：30 个地面站（GS）、30 个高空平台（HAP）、30 颗卫星（SAT）"
             "构成三层光网络，共 1978 条候选物理链路；地面站之间不断产生有密钥需求的通信请求。",
        "sz": 12, "c": DARK, "ls": 1.45}])
    draw_topology(s, X0, 4.35, X1 - X0)
    tb(s, X0, 16.7, X1 - X0, 1.2, [{
        "t": "物理图 = 三类节点 + 四类物理链路；逻辑图 = GS-GS 需求链路。两者共享节点集合，融合为一张全局图输入策略网络。",
        "sz": 10.5, "c": GRAY, "ls": 1.4}])


def page_definitions(deck):
    s = deck.page("关键定义与评价口径", "第一部分 · 场景分析与任务定义")
    rows = [
        ["术语", "含义"],
        ["时隙 t", "1 分钟。全年 525600 个时隙，状态、动作、奖励均以时隙为单位"],
        ["QKP 密钥池", "每条链路独立的密钥暂存区，有容量上限。链路生成的密钥先入池，再由路由分配去服务请求"],
        ["请求（Request）", "地面站对 (p, q) 之间的一条密钥需求，携带需求量、到达时刻、截止时间与优先级"],
        ["服务成功", "在截止时间前请求需求量被完整满足；路径受阻时按瓶颈跳部分服务，剩余量继续排队"],
        ["成功率", "按时完成服务的请求数 / 总请求数。标准验证取 5 个种子的均值"],
        ["切换半速", "链路从空闲恢复激活的首个时隙，生成速率 ×0.5（rate_decay_factor）"],
        ["双端口", "每个节点每时隙至多 1 条出边（Tx-out ≤ 1）与 1 条入边（Rx-in ≤ 1）"],
    ]
    draw_table(s, X0, BODY_TOP, [6.2, 23.87], rows, [1.0] + [1.55] * 7)
    tb(s, X0, 15.2, X1 - X0, 1.4, [{
        "t": "口径说明：成功率统一按「截止时间前完整满足的请求数 / 总请求数」统计；"
             "不同表格中的数字若协议不同（标准验证 / 固定场景）不可直接比较。",
        "sz": 10.5, "c": GRAY, "ls": 1.4}])


def page_constraints(deck):
    s = deck.page("约束建模与决策任务", "第一部分 · 场景分析与任务定义")
    section_head(s, "三类硬约束", X0, BODY_TOP)
    cards = [
        ("Tx-out ≤ 1", "每个节点每个时隙\n至多激活 1 条出边", "发送端单端口"),
        ("Rx-in ≤ 1", "每个节点每个时隙\n至多接收 1 条入边", "接收端单端口"),
        ("u→v 与 v→u 互斥", "同一对节点的两个方向\n不能同时激活", "避免自环冲突"),
    ]
    gap, cw = 0.6, (X1 - X0 - 1.2) / 3
    cx = X0
    for head, body, note in cards:
        rect(s, cx, 3.5, cw, 3.0, fill=WHITE, line=MID, lw=0.75)
        tb(s, cx + 0.3, 3.75, cw - 0.6, 0.7, [{"t": head, "sz": 13.5, "b": True, "c": BLACK}])
        tb(s, cx + 0.3, 4.5, cw - 0.6, 1.4, [{"t": body, "sz": 10.5, "c": DARK, "ls": 1.35}])
        tb(s, cx + 0.3, 5.85, cw - 0.6, 0.5, [{"t": note, "sz": 9.5, "c": GRAY}])
        cx += cw + gap

    section_head(s, "决策任务：每时隙输出一个全局匹配", X0, 7.05)
    box(s, X0, 7.95, 13.0, 2.1,
        "动作 = 一组互不冲突的有向弧\n（每个节点至多参与一进一出，未匹配节点保持空闲）",
        sz=11, fill=BLACK, tc=WHITE, b=True)
    seg(s, X0 + 13.3, 9.0, X0 + 14.6, 9.0, GRAY, 1.25)
    r1 = rect(s, X0 + 14.9, 7.95, 16.77, 2.1, fill=WHITE, line=MID, lw=0.75)
    tb(s, X0 + 15.2, 8.2, 16.2, 1.7, [{
        "t": "其余规则：一条请求只沿一条中继路径服务；中继链路按实际转发量扣减密钥池；"
             "切换链路在首个时隙以半速生成。掩码在构图前完成，策略只在合法动作上定义。",
        "sz": 10.5, "c": DARK, "ls": 1.4}])

    section_head(s, "与早期版本的差异", X0, 10.7)
    rows = [
        ["", "早期：节点独立采样", "当前：全局匹配"],
        ["动作生成", "每个节点独立采样一条边", "从全部合法有向弧中顺序采样一个匹配"],
        ["冲突消解", "由 resolver 事后贪心消解", "采样时即保证两端点空闲，无需消解"],
        ["优化对象", "提议分布 ≠ 环境执行的匹配", "联合 log-prob 就是环境执行的匹配"],
        ["结果", "成功率长期停在约 30%", "固定场景成功率 0.728 → 0.819"],
    ]
    draw_table(s, X0, 11.6, [5.0, 12.5, 12.57], rows, [0.95] + [1.05] * 4)


def page_bottleneck(deck):
    s = deck.page("问题的本质：瓶颈不在产能，而在可达性", "第一部分 · 场景分析与任务定义")
    cards = [
        ("4.7×10⁶", "密钥 / 时隙", "网络名义生成能力，远高于请求规模"),
        ("6–10×10⁴", "密钥 / 时隙", "地面站密钥请求的需求规模"),
        ("9.3%", "（链路, 时隙）非零", "仅有约 9.3% 的链路-时隙组合速率非零"),
        ("1.4%", "GS-SAT 可用槽", "稀疏性最严重的链路类型（HAP-SAT 6.8%）"),
    ]
    draw_kpi_cards(s, X0, BODY_TOP + 0.3, X1 - X0, cards)
    rect(s, X0, 9.6, X1 - X0, 2.4, fill=BLACK, line=None)
    tb(s, X0 + 0.6, 9.85, X1 - X0 - 1.2, 1.9, [{
        "t": "名义产能约为需求的 50 倍，但可见链路既稀疏又时变：生成得多不代表送得到，调度才决定密钥能否抵达请求方。",
        "sz": 13, "b": True, "c": WHITE, "ls": 1.45, "al": PP_ALIGN.CENTER}],
        anchor=MSO_ANCHOR.MIDDLE)
    section_head(s, "可用性由三重因素共同决定", X0, 12.6)
    rows = [
        ["因素", "作用", "典型表现"],
        ["视线遮挡（LOS）", "地球弦遮挡直接判定链路不可见", "GS-SAT 大部分时间不可见"],
        ["仰角阈值", "仰角 < 5° 的链路不予采用", "低轨卫星过境窗口外全部失效"],
        ["卫星轨道几何 / 天气", "决定可见时段与链路速率水平", "SAT-SAT 可用槽占比可达 51.2%"],
    ]
    draw_table(s, X0, 13.5, [6.5, 11.5, 12.07], rows, [0.95] + [1.0] * 3)


def page_channel_placeholder(deck):
    s = deck.page("信道建模与链路速率计算", "第二部分 · 信道建模")
    rect(s, X0, BODY_TOP + 0.2, X1 - X0, 1.5, fill=BG, line=None)
    tb(s, X0 + 0.5, BODY_TOP + 0.5, X1 - X0 - 1.0, 1.0, [{
        "t": "本部分内容待补充。以下为规划中的讲解框架，供后续填充。",
        "sz": 12, "b": True, "c": DARK}], anchor=MSO_ANCHOR.MIDDLE)
    rows = [
        ["计划小节", "拟覆盖内容"],
        ["速率生成链路", "物理量（距离 / 仰角 / 天气 / 云量）→ 离线预计算 k_max(bps) → H5 数据集 → 训练时查表"],
        ["协议层速率模型", "骨架效率 η 的构成与 GLLP / 诱骗态速率公式，关键参数取值"],
        ["可用性判定", "LOS 遮挡、仰角阈值、卫星过境窗口的判定方式"],
        ["切换半速与稀疏性", "切换时隙速率 ×0.5 的实现，以及分类型可用槽占比统计"],
    ]
    draw_table(s, X0, 4.6, [7.0, 23.07], rows, [1.0] + [1.35] * 4)
    tb(s, X0, 12.4, X1 - X0, 1.0, [{
        "t": "说明：速率在离线阶段用完整物理模型预计算并落盘为 H5，训练与评估阶段只做数组查表与归一化，不含实时物理计算。",
        "sz": 10.5, "c": GRAY, "ls": 1.4}])


def page_method_overview(deck):
    s = deck.page("方法总览：启发式预训练 + 强化学习微调", "第三部分 · 算法")
    draw_pipeline(s, X0, BODY_TOP + 0.4, X1 - X0)
    section_head(s, "为什么这样做", X0, 6.3)
    items = [
        ("专家起点", "随机策略成功率仅约 0.18，直接用 PPO 探索效率极低"),
        ("行为克隆", "克隆 PG-Phased 启发式动作，把初始化拉到 0.73–0.86 区间"),
        ("RL 微调", "从 BC 权重 warm-start，用 PPO 优化边打分策略，突破启发式上界"),
    ]
    cx = X0
    cw = (X1 - X0 - 1.2) / 3
    for head, body in items:
        rect(s, cx, 7.2, cw, 2.4, fill=WHITE, line=MID, lw=0.75)
        tb(s, cx + 0.35, 7.5, cw - 0.7, 0.7, [{"t": head, "sz": 13, "b": True, "c": BLACK}])
        tb(s, cx + 0.35, 8.35, cw - 0.7, 1.1, [{"t": body, "sz": 10.5, "c": DARK, "ls": 1.35}])
        cx += cw + 0.6

    section_head(s, "两个算法共用的核心概念：需求扩散重要性", X0, 10.3)
    tb(s, X0 + 0.42, 11.1, X1 - X0, 3.6, [
        {"t": "· 启发式用它给边打分、决定先开哪条路；强化学习把它作为物理边的一个输入特征（relay importance），"
              "让策略直接看到「这条边对满足当前需求有多重要」。", "sz": 11, "c": DARK, "ls": 1.45, "sp": 0},
        {"t": "· 这样做的收益：Actor 不需要自己从零学「哪些边能拼成一条路径」，只需判断「重要性高且速率、容量合适的边值不值得开」。",
         "sz": 11, "c": DARK, "ls": 1.45, "sp": 6},
    ])
    rect(s, X0, 15.2, X1 - X0, 2.0, fill=BG, line=None)
    tb(s, X0 + 0.6, 15.45, X1 - X0 - 1.2, 1.6, [{
        "t": "后续章节顺序：启发式（2 页）→ 强化学习的建模、状态、网络、动作、奖励、训练（6 页）。",
        "sz": 11, "c": DARK}], anchor=MSO_ANCHOR.MIDDLE)


def page_heuristic_importance(deck):
    s = deck.page("启发式①：BFS 需求扩散（relay importance）", "第三部分 · 算法")
    tb(s, X0, BODY_TOP, X1 - X0, 1.2, [{
        "t": "目标：把所有待处理请求的压力，压缩成「每条物理链路一个标量」，回答「哪条边最值得开」。",
        "sz": 11.5, "b": True, "c": DARK, "ls": 1.4}])

    steps = [
        ("① 构建可见图", "以当前合法可见的物理边构成图 G_t，待处理请求对 (p, q) 构成需求边"),
        ("② 计算紧迫度加权需求", "剩余量 × exp(已等待时长 / τ)，τ = 截止时长 × 0.8，越接近截止压力越大"),
        ("③ BFS 求最短跳数", "以 p、q 为起点求到各节点的最短跳数，得到 a = min(d(p,u), d(p,v))、b 同理"),
        ("④ 累积到边上", "若 a + b + 1 ≤ K（K = 3 跳），按跳数衰减 η 与剩余容量加权累加到边"),
        ("⑤ 归一化", "除以本时隙最大累积值，得到 [0, 1] 的重要性 I_i"),
    ]
    y = 3.9
    for head, body in steps:
        rect(s, X0, y, 0.14, 0.95, fill=BLACK, rounded=False)
        tb(s, X0 + 0.45, y - 0.05, 6.0, 0.6, [{"t": head, "sz": 11.5, "b": True, "c": BLACK}])
        tb(s, X0 + 6.6, y - 0.05, X1 - X0 - 6.6, 1.0, [{"t": body, "sz": 10.5, "c": DARK, "ls": 1.35}])
        y += 1.25

    rect(s, X0, 10.3, X1 - X0, 3.3, fill=BLACK, line=None)
    tb(s, X0 + 0.7, 10.6, X1 - X0 - 1.4, 2.8, [
        {"t": "紧迫度加权剩余需求：B_pq(t) = Σ 剩余量 × exp(age / τ)　　　τ = deadline_length × 0.8",
         "sz": 12, "c": WHITE, "ls": 1.5},
        {"t": "边重要性累积：relay_i += B_pq × η^(a+b−1) × cap_left_i　　　η = 0.25（跳数衰减）",
         "sz": 12, "c": WHITE, "ls": 1.5, "sp": 8},
        {"t": "归一化：I_i = relay_i / max_j(relay_j) ∈ [0, 1]",
         "sz": 12, "c": WHITE, "ls": 1.5, "sp": 8},
    ])
    tb(s, X0, 14.0, X1 - X0, 2.2, [
        {"t": "工程取舍：实现中允许「当前不可用但仍持有密钥库存的链路」参与 BFS，"
              "使刚停用的链路不会立刻失去存在感；重要性只赋给当前合法可选的边。", "sz": 10.5, "c": GRAY, "ls": 1.4},
        {"t": "K = 3 跳的含义：一条物理边要成为某请求对的候选中继，两端地面站经过它连通的总跳数不超过 3。",
         "sz": 10.5, "c": GRAY, "ls": 1.4, "sp": 5},
    ])


def page_heuristic_phased(deck):
    s = deck.page("启发式②：分阶段路径调度（PG-Phased）", "第三部分 · 算法")
    tb(s, X0, BODY_TOP, X1 - X0, 1.0, [{
        "t": "有了边打分只解决了「谁值得开」，还需要决定「按什么顺序、把整条路开出来」。",
        "sz": 11.5, "b": True, "c": DARK}])
    stages = [
        ("① 存量密钥网络", "优先用已持有的密钥库存直接服务请求，不激活任何新链路"),
        ("② 混用库存的辅助路径", "库存不足以完整服务时，用库存 + 新生成混合补足"),
        ("③ 全新路径", "库存不可用时，激活整条中继路径生成新密钥服务"),
    ]
    gap = 0.6
    cw = (X1 - X0 - gap * 2) / 3
    cx = X0
    for head, body in stages:
        rect(s, cx, 3.9, cw, 2.6, fill=WHITE, line=MID, lw=0.75)
        rect(s, cx, 3.9, cw, 0.16, fill=BLACK, rounded=False)
        tb(s, cx + 0.35, 4.35, cw - 0.7, 0.7, [{"t": head, "sz": 13, "b": True, "c": BLACK}])
        tb(s, cx + 0.35, 5.2, cw - 0.7, 1.2, [{"t": body, "sz": 10.5, "c": DARK, "ls": 1.35}])
        cx += cw + gap
    tb(s, X0 + 0.1, 6.75, X1 - X0, 0.8, [{
        "t": "配合 ServeProbe 路由：按瓶颈跳做部分服务，避免一条路径因中间某一跳受限而整体白开。",
        "sz": 11, "c": GRAY, "ls": 1.4}])

    section_head(s, "效果与定位（实测）", X0, 7.9)
    rows = [
        ["策略", "协议", "成功率"],
        ["path_score_greedy_phased（本启发式）", "标准验证（5 种子，确定性）", "0.8104 ± 0.0575"],
        ["path_score_greedy_phased（本启发式）", "固定场景（12 种子）", "0.7765"],
        ["greedy_relay（第二名非学习方法）", "标准验证", "0.435"],
        ["随机策略", "—", "≈ 0.18"],
    ]
    draw_table(s, X0, 8.8, [12.0, 12.0, 8.07], rows, [1.0] + [1.05] * 4,
               align=[PP_ALIGN.LEFT, PP_ALIGN.LEFT, PP_ALIGN.CENTER])
    rect(s, X0, 14.5, X1 - X0, 2.5, fill=BG, line=None)
    tb(s, X0 + 0.6, 14.8, X1 - X0 - 1.2, 2.0, [
        {"t": "本启发式是最强的非学习方法，比第二名高 1.86 倍。", "sz": 11.5, "b": True, "c": BLACK},
        {"t": "它同时承担另一个角色：作为强化学习的行为克隆专家，把可复现的调度经验注入策略网络。",
         "sz": 11, "c": DARK, "sp": 6},
    ])


def page_rl_mdp(deck):
    s = deck.page("强化学习建模：带图结构的 MDP", "第三部分 · 算法")
    rows = [
        ["MDP 元素", "定义"],
        ["状态 s_t", "物理图与逻辑图融合的全局图：节点集合 + 物理链路边 + GS-GS 需求边，每个实体携带特征与历史"],
        ["动作 a_t", "一个全局匹配：从合法物理边中选出一组互不相交的有向弧，未匹配节点保持空闲"],
        ["转移 P", "由链路速率的时间演化（天气、轨道几何）、密钥生成 / 过期、请求到达 / 服务 / 过期共同决定"],
        ["奖励 r_t", "B1 shaped 稠密奖励：服务成功 + 重要性加权生成 − 失败 / 过期 − 切换惩罚"],
        ["折扣因子 γ", "0.99（GAE 平滑系数 λ = 0.95）"],
    ]
    draw_table(s, X0, BODY_TOP, [5.5, 24.57], rows, [1.0] + [1.75] * 5)

    section_head(s, "图的两个关键性质", X0, 13.0)
    items = [
        ("实时动态", "物理图只保留当前合法的链路，逻辑图只保留活跃或近期出现过的请求对，图结构每个时隙都在变。"),
        ("动作掩码前置", "不可用、速率低于下限、被禁用类型的链路在构图前就被剔除，根本不会出现在候选集合里；空闲动作恒合法。"),
    ]
    y = 13.9
    for head, body in items:
        rect(s, X0, y, 0.14, 0.95, fill=BLACK, rounded=False)
        tb(s, X0 + 0.45, y - 0.05, 4.4, 0.6, [{"t": head, "sz": 11.5, "b": True, "c": BLACK}])
        tb(s, X0 + 5.0, y - 0.08, X1 - X0 - 5.0, 1.1, [{"t": body, "sz": 10.5, "c": DARK, "ls": 1.35}])
        y += 1.3


def page_rl_state(deck):
    s = deck.page("状态表示：三类实体的特征", "第三部分 · 算法")
    tb(s, X0, BODY_TOP, X1 - X0, 0.9, [{
        "t": "状态由三类实体的特征拼成，全部在送入网络前做归一化。",
        "sz": 11.5, "b": True, "c": DARK}])
    cols = [
        ("节点特征", [
            "节点类型 one-hot（GS / HAP / SAT）",
            "关联链路密钥池总库存 / 总容量",
            "关联链路剩余容量占比",
            "以该节点为源 / 目的的需求量",
            "队列压力：仍在等待服务的请求数",
            "近期到达量（窗口 15 / 60 / 240 分钟）",
            "是否存在可用相邻链路",
            "时间周期特征（分钟、年内的 sin/cos 编码）",
        ]),
        ("物理边特征", [
            "链路类型 one-hot（4 类）",
            "当前可用性、当前归一化速率",
            "未来 H = 6 步速率与可用性（前向窗口）",
            "窗口内速率增量 / 均值 / 最大值",
            "上一时隙是否激活",
            "relay importance（需求扩散重要性）",
            "剩余 QKP 容量占比",
        ]),
        ("需求边特征", [
            "该请求对的 pending 密钥量",
            "pending 请求数",
            "最小 / 平均剩余截止时间",
            "平均已等待时长",
            "请求量 × 优先级之和",
            "等待时间分桶 × 10（各桶剩余需求量）",
        ]),
    ]
    gap = 0.6
    cw = (X1 - X0 - gap * 2) / 3
    cx = X0
    for head, bullets in cols:
        rect(s, cx, 3.55, cw, 0.85, fill=BLACK, line=None)
        tb(s, cx + 0.3, 3.68, cw - 0.6, 0.6, [{"t": head, "sz": 12.5, "b": True, "c": WHITE,
                                              "al": PP_ALIGN.CENTER}], anchor=MSO_ANCHOR.MIDDLE)
        rect(s, cx, 4.4, cw, 8.6, fill=WHITE, line=MID, lw=0.75)
        paras = [{"t": "· " + b, "sz": 10, "c": DARK, "ls": 1.3, "sp": 0 if i == 0 else 7}
                 for i, b in enumerate(bullets)]
        tb(s, cx + 0.35, 4.7, cw - 0.7, 8.0, paras)
        cx += cw + gap

    section_head(s, "两个设计要点", X0, 13.35)
    tb(s, X0 + 0.45, 14.2, X1 - X0 - 0.45, 3.0, [
        {"t": "· 未来速率进入特征：离线数据中未来速率已知，策略可以据此规划「何时激活、何时让链路休息」，"
              "而不是只看当前这一分钟。", "sz": 10.5, "c": DARK, "ls": 1.4},
        {"t": "· 等待分桶把截止时长均匀切成 10 个桶，桶内记录该等待区间仍剩余的需求量，"
              "使网络不仅知道「有多少需求」，还知道「等了多久、还差多久到期」。", "sz": 10.5, "c": DARK, "ls": 1.4, "sp": 6},
        {"t": "· 历史由共享 LSTM 编码（默认关闭）：只编码过去，未来信息仅通过前向窗口进入网络。",
         "sz": 10.5, "c": DARK, "ls": 1.4, "sp": 6},
    ])


def page_rl_network(deck):
    s = deck.page("网络结构：共享编码器 + 边打分 Actor + 全局 Critic", "第三部分 · 算法")
    stages = [
        ("特征拼接", "手工特征\n(+ LSTM 历史)"),
        ("图编码器 GNN", "边条件消息传递\n3 层 × 128 维\nLayerNorm + 残差"),
        ("节点 / 边嵌入", "每条物理边的\n统一表示"),
        ("Actor：边打分", "MLP(h_u, h_v, e_uv)\n→ 每条合法弧一个分数"),
        ("Critic：全局价值", "typed-mean 池化\n+ 规模计数 → V(s)"),
    ]
    bh, gap = 3.0, 0.55
    bw = (X1 - X0 - gap * 4) / 5
    cx = X0
    for i, (head, body) in enumerate(stages):
        fill = BLACK if i in (1, 3) else WHITE
        tc = WHITE if i in (1, 3) else DARK
        rect(s, cx, BODY_TOP + 0.5, bw, bh, fill=fill, line=(None if fill == BLACK else MID), lw=0.75)
        tb(s, cx + 0.2, BODY_TOP + 0.85, bw - 0.4, 0.8,
           [{"t": head, "sz": 12, "b": True, "c": tc, "al": PP_ALIGN.CENTER}])
        tb(s, cx + 0.2, BODY_TOP + 1.75, bw - 0.4, 1.5,
           [{"t": body, "sz": 10, "c": tc, "al": PP_ALIGN.CENTER, "ls": 1.35}])
        if i < 4:
            seg(s, cx + bw + 0.1, BODY_TOP + 2.0, cx + bw + gap - 0.1, BODY_TOP + 2.0, GRAY, 1.25)
        cx += bw + gap

    section_head(s, "三个关键设计选择", X0, 7.1)
    rows = [
        ["设计", "做法与理由"],
        ["共享编码器 + 共享 Actor",
         "所有节点、所有边共享同一套参数，属于 MAPPO 在单图场景下的等价形式；Critic 是全局价值函数"],
        ["demand_edge 模式（默认）",
         "只让需求边更新节点嵌入，物理边的能力信息通过 relay importance 直接压进边特征，避免「高速率但与需求无关」的链路污染节点状态"],
        ["typed-mean 池化 + 规模计数",
         "节点 / 物理边 / 需求边分别取平均再拼接，避免两类边相互稀释；补上 log 规模计数弥补均值池化丢失的图规模信息"],
    ]
    draw_table(s, X0, 8.0, [8.0, 22.07], rows, [1.0] + [1.6] * 3)
    rect(s, X0, 15.0, X1 - X0, 2.2, fill=BG, line=None)
    tb(s, X0 + 0.6, 15.25, X1 - X0 - 1.2, 1.8, [{
        "t": "边嵌入在 GNN 中不做逐层更新，保持与输入特征一一对应；决策最终落在链路上，"
             "因此 Actor 只需学会判断「重要性高、速率和容量合适的边值不值得开」。",
        "sz": 11, "c": DARK, "ls": 1.4}], anchor=MSO_ANCHOR.MIDDLE)


def page_rl_action(deck):
    s = deck.page("动作设计：全局匹配（关键设计）", "第三部分 · 算法")
    tb(s, X0, BODY_TOP, X1 - X0, 0.9, [{
        "t": "这是整个项目里返工最多、也最关键的一处设计。",
        "sz": 12, "b": True, "c": DARK}])

    half = (X1 - X0 - 1.2) / 2
    # 左：早期做法
    rect(s, X0, 3.5, half, 6.4, fill=WHITE, line=MID, lw=0.75)
    rect(s, X0, 3.5, half, 0.7, fill=BG, rounded=False)
    tb(s, X0 + 0.4, 3.62, half - 0.8, 0.55, [{"t": "早期做法（已废弃）", "sz": 12.5, "b": True, "c": DARK}])
    tb(s, X0 + 0.4, 4.45, half - 0.8, 5.2, [
        {"t": "· 每个节点独立采样一条边", "sz": 11, "c": DARK, "ls": 1.4},
        {"t": "· 采样结果由 resolver 事后贪心消解冲突", "sz": 11, "c": DARK, "ls": 1.4, "sp": 7},
        {"t": "· 问题的根源：PPO 优化的是「提议分布」，而环境实际执行的是消解后的匹配，两者不一致",
         "sz": 11, "c": DARK, "ls": 1.4, "sp": 7},
        {"t": "· 后果：成功率长期停在约 30%，怎么调参都上不去", "sz": 11, "c": DARK, "ls": 1.4, "sp": 7},
    ])
    # 右：当前做法
    rect(s, X0 + half + 1.2, 3.5, half, 6.4, fill=BLACK, line=None)
    tb(s, X0 + half + 1.6, 3.75, half - 0.8, 0.6, [{"t": "当前做法：全局匹配", "sz": 12.5, "b": True, "c": WHITE}])
    tb(s, X0 + half + 1.6, 4.55, half - 0.8, 5.2, [
        {"t": "· Actor 对每条合法有向弧 (tx_target, rx_source) 打分", "sz": 11, "c": WHITE, "ls": 1.4},
        {"t": "· 从全部合法弧出发，每一步只在「两端点仍空闲」的弧中按 softmax 采样一条（含 STOP 选项），直到没有可用弧",
         "sz": 11, "c": WHITE, "ls": 1.4, "sp": 7},
        {"t": "· 关键性质：采样出来的匹配，就是环境真正执行的匹配", "sz": 11, "c": WHITE, "ls": 1.4, "sp": 7},
        {"t": "· PPO 优化的正是该匹配的联合 log-prob（向量化实现）", "sz": 11, "c": WHITE, "ls": 1.4, "sp": 7},
    ])

    box(s, X0, 10.3, X1 - X0, 1.6,
        "π_θ(M | s_t) =  softmax(s_e / T)，e_k 为第 k 步选中弧，A_k 为第 k 步两端点仍空闲的合法弧集合，温度 T 默认 1.0",
        sz=11.5, fill=BG, tc=DARK)

    section_head(s, "修复后的效果", X0, 12.3)
    rows = [
        ["版本", "动作层机制", "固定场景成功率"],
        ["早期", "节点独立采样 + resolver 贪心消解", "长期停在约 0.30"],
        ["当前", "全局匹配顺序采样，采样即执行", "0.728 → 0.819"],
    ]
    draw_table(s, X0, 13.2, [5.0, 15.0, 10.07], rows, [1.0] + [1.15] * 2,
               align=[PP_ALIGN.LEFT, PP_ALIGN.LEFT, PP_ALIGN.CENTER])
    tb(s, X0, 16.6, X1 - X0, 1.0, [{
        "t": "结论：动作层与优化目标是否一致，比调参更决定成败。",
        "sz": 11.5, "b": True, "c": BLACK}])


def page_rl_reward(deck):
    s = deck.page("奖励设计：B1 shaped 稠密奖励", "第三部分 · 算法")
    rect(s, X0, BODY_TOP, X1 - X0, 2.4, fill=BLACK, line=None)
    tb(s, X0 + 0.7, BODY_TOP + 0.35, X1 - X0 - 1.4, 1.8, [
        {"t": "r_t =  w_s · S_t / R_served　+　w_g · Σ(added_i · I_i − w_low · added_i · (1 − I_i)) / Σ added_i",
         "sz": 12.5, "c": WHITE, "ls": 1.5},
        {"t": "　　　　−　(w_f · F_t + w_e · E_t) / R_served　−　w_c · C_t",
         "sz": 12.5, "c": WHITE, "ls": 1.5, "sp": 6},
    ])
    rows = [
        ["分量", "含义", "权重"],
        ["服务成功 S_t", "本步成功服务的密钥量", "w_s = 20.0"],
        ["重要性加权生成", "生成在重要中继路径上且实际入池的密钥份额（占比形式）", "w_g = 0.02"],
        ["低重要性惩罚", "存入低重要性链路的部分被按比例扣分", "w_low = 0.01"],
        ["失败 / 过期", "失败密钥量（含过期请求）与过期密钥量", "w_f = w_e = 0.01"],
        ["切换惩罚", "本步新激活（上一时隙未激活）的链路条数", "w_c = 0.001"],
        ["需求参考值", "服务侧与失败侧统一除以的固定参考值", "R_served = 100000"],
    ]
    draw_table(s, X0, 5.4, [6.5, 18.5, 5.07], rows, [1.0] + [1.15] * 6)

    section_head(s, "三个设计要点", X0, 13.7)
    tb(s, X0 + 0.45, 14.5, X1 - X0 - 0.45, 3.2, [
        {"t": "· 固定参考值：需求侧统一除以固定的 R_served，保证「同样 1 bit 服务量」在开局、"
              "深夜、高峰获得相同的奖励尺度，避免奖励在 episode 内部随时间漂移。", "sz": 10.5, "c": DARK, "ls": 1.4},
        {"t": "· 占比形式：供应侧取「重要性加权和占本步存入总量的比例」，取值范围约在 [−w_g·w_low, w_g]，"
              "只奖励真正生成在重要中继路径上的密钥份额，避免大额原始生成主导奖励。", "sz": 10.5, "c": DARK, "ls": 1.4, "sp": 6},
        {"t": "· 切换惩罚按条数计：不再除以百万级参考值，量级与其它分量可比。",
         "sz": 10.5, "c": DARK, "ls": 1.4, "sp": 6},
    ])


def page_rl_training(deck):
    s = deck.page("训练流程：多进程 rollout → GAE → PPO", "第三部分 · 算法")
    steps = [
        ("① 多进程 rollout", "worker 进程池并行采样，模型权重与探索温度随任务下发，保证各 worker 的探索温度同步；"
                              "每步记录观测、动作、旧策略 log-prob、价值、奖励、终止标志"),
        ("② GAE 优势估计", "δ_t = r_t + γ(1−d_t)V(s_{t+1}) − V(s_t)；A_t = δ_t + γλ(1−d_t)A_{t+1}；"
                            "回报 R̂_t = A_t + V(s_t)，γ = 0.99、λ = 0.95"),
        ("③ PPO 更新", "优势标准化 → 裁剪 surrogate（ε = 0.2）→ 价值损失（系数 0.5）+ 熵正则 → "
                        "梯度裁剪 ‖g‖₂ ≤ 0.5 → Adam"),
        ("④ 早停与保存", "minibatch 平均 KL 散度 > 0.03 时结束本轮 epoch；定期保存含优化器状态的 checkpoint，支持断点续训"),
    ]
    y = BODY_TOP
    for head, body in steps:
        rect(s, X0, y, 0.14, 1.35, fill=BLACK, rounded=False)
        tb(s, X0 + 0.5, y - 0.08, 6.3, 0.6, [{"t": head, "sz": 12, "b": True, "c": BLACK}])
        tb(s, X0 + 7.0, y - 0.1, X1 - X0 - 7.0, 1.5, [{"t": body, "sz": 10.5, "c": DARK, "ls": 1.4}])
        y += 1.85

    section_head(s, "关键超参数", X0, 10.2)
    rows = [
        ["参数", "取值", "参数", "取值"],
        ["隐藏维 / GNN 层数", "128 / 3 层", "折扣因子 γ", "0.99"],
        ["未来速率窗口 H", "6 步", "GAE 系数 λ", "0.95"],
        ["需求等待分桶", "10 个", "PPO 裁剪 ε", "0.2"],
        ["学习率（Actor / Critic）", "3e−4 / 1e−3", "梯度裁剪 / target KL", "0.5 / 0.03"],
    ]
    draw_table(s, X0, 11.1, [8.5, 6.85, 8.5, 6.22], rows, [0.95] + [1.0] * 4)
    tb(s, X0, 16.4, X1 - X0, 1.2, [{
        "t": "训练模式：continuous（随机起点连续训练）、fixed_day（固定日跑满 1440 步）、curriculum（短局起步逐步加长）。",
        "sz": 10.5, "c": GRAY, "ls": 1.4}])


def page_results(deck):
    s = deck.page("评测协议与结果对比", "第四部分 · 结果与存在的问题")
    section_head(s, "两套评测协议", X0, BODY_TOP)
    tb(s, X0 + 0.45, BODY_TOP + 0.85, 14.6, 2.4, [
        {"t": "· 标准验证：5 个随机种子、确定性评估（argmax），用于横向比较不同方法。", "sz": 10.5, "c": DARK, "ls": 1.4},
        {"t": "· 固定场景：day 0 + 固定请求种子、12 种子，用于调试与验证奖励设计（可复现）。", "sz": 10.5, "c": DARK, "ls": 1.4, "sp": 6},
    ])
    tb(s, X0 + 15.6, BODY_TOP + 0.85, X1 - X0 - 15.6, 2.4, [
        {"t": "· 同口径评估：checkpoint 用与训练指标完全相同的口径复算，避免「训练好、评估差」。",
         "sz": 10.5, "c": DARK, "ls": 1.4},
        {"t": "注意：不同协议的绝对数值不可直接比较，表中已标注协议。", "sz": 10.5, "c": GRAY, "ls": 1.4, "sp": 6},
    ])
    rows = [
        ["策略", "协议", "成功率"],
        ["启发式 path_score_greedy_phased", "标准验证（5 种子，确定性）", "0.8104 ± 0.0575"],
        ["启发式 path_score_greedy_phased", "固定场景（12 种子）", "0.7765"],
        ["BC 预热权重（克隆启发式）", "标准验证（5 种子，确定性）", "0.7581"],
        ["BC 预热权重", "固定场景（12 种子，确定性）", "0.7280"],
        ["RL —— 固定场景训练 20 轮", "固定场景", "0.819"],
        ["RL —— 全局训练 30 轮", "标准验证", "0.713 ± 0.033"],
        ["随机策略", "—", "≈ 0.18"],
    ]
    draw_table(s, X0, 5.75, [12.0, 12.0, 8.07], rows, [1.0] + [1.2] * 7,
               align=[PP_ALIGN.LEFT, PP_ALIGN.LEFT, PP_ALIGN.CENTER])
    rect(s, X0, 15.75, X1 - X0, 1.9, fill=BLACK, line=None)
    tb(s, X0 + 0.6, 15.98, X1 - X0 - 1.2, 1.5, [{
        "t": "一句话总结：固定场景下 RL 已经超过启发式；但同样的方法搬到全局训练上，提升消失了。",
        "sz": 12.5, "b": True, "c": WHITE, "al": PP_ALIGN.CENTER}], anchor=MSO_ANCHOR.MIDDLE)


def page_result_reading(deck):
    s = deck.page("结果解读：BC 的价值与固定场景的突破", "第四部分 · 结果与存在的问题")
    cards = [
        ("0.8104 → 0.7581", "行为克隆的代价", "克隆专家动作会损失约 5.2 个百分点，但把 RL 的初始成功率从「随机 ≈ 0.18」抬升到「0.73–0.86」区间"),
        ("0.728 → 0.819", "固定场景的突破", "在 day 0 + 固定请求种子的场景下训练 20 轮，成功率超过启发式基线 0.7765"),
        ("0.713 ± 0.033", "全局训练的停滞", "标准验证 5 种子 6 次评估均值与纯噪声波动一致，低于 BC 起点与启发式"),
    ]
    gap = 0.6
    cw = (X1 - X0 - gap * 2) / 3
    cx = X0
    for value, label, note in cards:
        rect(s, cx, BODY_TOP + 0.2, cw, 7.4, fill=WHITE, line=MID, lw=0.75)
        rect(s, cx, BODY_TOP + 0.2, cw, 0.16, fill=BLACK, rounded=False)
        tb(s, cx + 0.4, BODY_TOP + 1.4, cw - 0.8, 1.2,
           [{"t": value, "sz": 19, "b": True, "c": BLACK, "al": PP_ALIGN.CENTER}])
        tb(s, cx + 0.4, BODY_TOP + 2.8, cw - 0.8, 0.7,
           [{"t": label, "sz": 12, "b": True, "c": DARK, "al": PP_ALIGN.CENTER}])
        hline(s, cx + 1.2, BODY_TOP + 3.65, cw - 2.4, LIGHT, 0.75)
        tb(s, cx + 0.5, BODY_TOP + 3.95, cw - 1.0, 3.2,
           [{"t": note, "sz": 10.5, "c": GRAY, "al": PP_ALIGN.CENTER, "ls": 1.4}])
        cx += cw + gap

    section_head(s, "可以确认的三件事", X0, 10.7)
    tb(s, X0 + 0.45, 11.6, X1 - X0 - 0.45, 5.4, [
        {"t": "① 行为克隆是有价值的：它把 RL 从「随机探索」直接抬到接近专家水平，使固定场景的微调变得可行。",
         "sz": 11.5, "c": DARK, "ls": 1.45},
        {"t": "② 模型与奖励设计在固定场景下是有效的：同一套网络和奖励能让成功率超过启发式，说明结构和奖励信号没有方向性错误。",
         "sz": 11.5, "c": DARK, "ls": 1.45, "sp": 9},
        {"t": "③ 问题不在价值估计：训练全程 critic 与真实回报的相关系数保持在 0.89，说明 Critic 学得准，瓶颈在策略更新侧。",
         "sz": 11.5, "c": DARK, "ls": 1.45, "sp": 9},
    ])
    rect(s, X0, 16.2, X1 - X0, 1.6, fill=BG, line=None)
    tb(s, X0 + 0.6, 16.4, X1 - X0 - 1.2, 1.2, [{
        "t": "于是问题被收敛成一个：为什么「固定场景有效」的更新方式，在全局训练里不再产生增益？",
        "sz": 12, "b": True, "c": BLACK}], anchor=MSO_ANCHOR.MIDDLE)


def page_problem(deck):
    s = deck.page("存在的问题：全局训练成功率停滞", "第四部分 · 结果与存在的问题")
    section_head(s, "现象", X0, BODY_TOP)
    rows = [
        ["观察项", "数据"],
        ["标准验证 6 次评估", "0.706 / 0.715 / 0.759 / 0.711 / 0.729 / 0.659，均值 0.713 ± 0.033"],
        ["与起点的关系", "低于 BC 起点 0.758，也低于启发式 0.810"],
        ["波动性质", "与「纯噪声围绕 0.713 波动」一致，观察不到上升趋势"],
        ["训练侧指标", "8 局采样均值 0.858 → 0.842，reward 92 → 91，全程持平"],
        ["策略熵", "3.33 → 4.01，持续向随机化漂移"],
        ["固定场景对照", "smoke 训练 60 轮后同口径评估 0.8574，反而低于 BC 基线 0.8622"],
    ]
    draw_table(s, X0, 3.55, [7.5, 22.57], rows, [1.0] + [1.05] * 6)

    section_head(s, "诊断：问题被定位在 Actor 侧", X0, 11.4)
    items = [
        ("Critic 全程健康", "corr(V, R) = 0.89，价值估计没有崩塌，因此不是「信号不可信」导致的学不动。"),
        ("策略熵持续上升", "从 3.33 升到 4.01，说明策略在向高熵、随机化方向漂移，而不是收敛到更确定的调度规则。"),
        ("固定场景同样退化", "60 轮后同口径评估低于 BC 基线，说明继续 PPO 反而被高熵样本轻微污染。"),
    ]
    y = 12.3
    for head, body in items:
        rect(s, X0, y, 0.14, 0.9, fill=BLACK, rounded=False)
        tb(s, X0 + 0.45, y - 0.05, 6.4, 0.6, [{"t": head, "sz": 11.5, "b": True, "c": BLACK}])
        tb(s, X0 + 7.0, y - 0.08, X1 - X0 - 7.0, 1.1, [{"t": body, "sz": 10.5, "c": DARK, "ls": 1.35}])
        y += 1.35


def page_next_steps(deck):
    s = deck.page("原因假设与下一步计划", "第四部分 · 结果与存在的问题")
    section_head(s, "三条待验证的假设", X0, BODY_TOP)
    rows = [
        ["假设", "依据", "验证方式"],
        ["探索温度过高（1.2 起步）把策略推向高熵",
         "策略熵从 3.33 升到 4.01",
         "降低探索温度后，观察熵与成功率的联合变化"],
        ["PPO 每轮更新量太小，策略学不动",
         "2880 步 ÷ minibatch 1024 ≈ 3 次梯度 / 轮",
         "减小 minibatch 或增大 epoch 数，提高每轮更新量"],
        ["BC 起点已接近局部最优，继续 PPO 被高熵样本污染",
         "固定场景 60 轮后同口径评估反低于 BC 基线",
         "降低学习率、加强 KL 约束后同口径复验"],
    ]
    draw_table(s, X0, 3.55, [11.0, 9.0, 10.07], rows, [1.0] + [1.5] * 3)

    section_head(s, "下一步计划", X0, 9.2)
    plans = [
        ("① 参数调整与复验", "按上表逐项调整训练参数，全部用同口径评估验证，确认是否恢复上升趋势"),
        ("② 训练配置统一", "统一 episode_steps 与 rollout_steps 为 360，避免 GAE 在轨迹截断处产生偏差，同时降低算力负担"),
        ("③ 补齐对比基线", "完成信道建模部分，并推进与 MILP 全局最优解、启发式上界的定量对比"),
        ("④ 加速训练循环", "缓存静态图结构、向量化最短路径计算，压缩单轮 rollout-update 的耗时"),
    ]
    y = 10.1
    for head, body in plans:
        rect(s, X0, y, 0.14, 1.15, fill=BLACK, rounded=False)
        tb(s, X0 + 0.5, y - 0.05, 6.6, 0.6, [{"t": head, "sz": 11.5, "b": True, "c": BLACK}])
        tb(s, X0 + 7.3, y - 0.08, X1 - X0 - 7.3, 1.3, [{"t": body, "sz": 10.5, "c": DARK, "ls": 1.35}])
        y += 1.6

    rect(s, X0, 16.6, X1 - X0, 1.6, fill=BLACK, line=None)
    tb(s, X0 + 0.6, 16.8, X1 - X0 - 1.2, 1.2, [{
        "t": "核心判断：动作层设计已经打通（固定场景可超启发式），剩下的问题是「如何在高熵全局探索下保住已学到的策略」。",
        "sz": 12, "b": True, "c": WHITE, "al": PP_ALIGN.CENTER}], anchor=MSO_ANCHOR.MIDDLE)


def page_thanks(prs, blank, sources):
    slide = clone(prs, blank, sources["end"])
    for shp in slide.shapes:
        if not shp.has_text_frame:
            continue
        t = shp.text_frame.text.strip()
        if "MINIMAL STYLE" in t:
            set_text(shp, "组会进展汇报", align=PP_ALIGN.CENTER)
        elif t == "极简风":
            set_text(shp, "谢谢观看", align=PP_ALIGN.CENTER)
        elif t.startswith("谢谢观看"):
            set_text(shp, "QKD-SAGIN 生产端调度", align=PP_ALIGN.CENTER)
    return slide


# ============================================================ 主流程
def fix_theme_font(path, old="造字工房悦黑体验版纤细体", new=FONT):
    tmp = path + ".tmp"
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename.startswith("ppt/theme/") and item.filename.endswith(".xml"):
                data = data.decode("utf-8").replace(old, new).encode("utf-8")
            zout.writestr(item, data)
    os.replace(tmp, path)


def main():
    prs, blank, content, sources = load_template()

    # 封面 / 目录
    build(prs, blank, content, sources)

    # 为保持章节顺序，这里按最终顺序重建：清空后按 Part 顺序逐页生成
    # （build 已放置封面与目录；过渡页在以下函数中按序插入）
    deck = Deck(prs, content)
    deck.n = 2

    def section(idx, title, desc):
        s = clone(prs, blank, sources["sec"][idx])
        for shp in s.shapes:
            if not shp.has_text_frame:
                continue
            t = shp.text_frame.text.strip()
            if t == "请在此输入标题":
                set_text(shp, title)
            elif t.startswith("your content"):
                set_text(shp, desc)
        deck.n += 1

    section(0, "场景分析与任务定义",
            "QKD-SAGIN 三层 FSO 网络 / 关键定义 / 双端口约束下的决策任务")
    page_scenario(deck)
    page_definitions(deck)
    page_constraints(deck)
    page_bottleneck(deck)

    section(1, "信道建模与链路速率（待补充）",
            "速率生成链路 / 协议层模型 / 可用性判定与稀疏性")
    page_channel_placeholder(deck)

    section(2, "算法：启发式与强化学习",
            "BFS 需求扩散启发式 / Graph-MAPPO / 全局匹配动作与奖励设计")
    page_method_overview(deck)
    page_heuristic_importance(deck)
    page_heuristic_phased(deck)
    page_rl_mdp(deck)
    page_rl_state(deck)
    page_rl_network(deck)
    page_rl_action(deck)
    page_rl_reward(deck)
    page_rl_training(deck)

    section(3, "结果与存在的问题",
            "评测协议 / 各算法结果对比 / 全局训练停滞与诊断 / 下一步计划")
    page_results(deck)
    page_result_reading(deck)
    page_problem(deck)
    page_next_steps(deck)

    page_thanks(prs, blank, sources)

    prs.core_properties.title = "QKD-SAGIN 生产端密钥生成链路调度 · 组会进展汇报"
    prs.core_properties.author = "qkd_rl"
    prs.save(OUTPUT)
    fix_theme_font(OUTPUT)
    print("生成完成：%s，共 %d 页" % (OUTPUT, len(prs.slides._sldIdLst)))


if __name__ == "__main__":
    main()