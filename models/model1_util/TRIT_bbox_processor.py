# -*- coding: utf-8 -*-
"""
CRAFT polygon / rotated-box → axis-aligned rectangle 변환 + 인접 박스 병합 유틸
Author  : Jangyeon-Kim project
Created : 2025-05-23
"""

import os
import cv2
import numpy as np
from .craft_utils import getBlockBoxes

# ---------------------------------------------------------------------------
# 1) 4-점(또는 N-점) → (x1, y1, x2, y2) 축정렬 직사각형
# ---------------------------------------------------------------------------
def to_rect(pts):
    """
    pts : np.ndarray([[x, y], ...])     ─ shape (4,2) or (N,2)
    return : [x1, y1, x2, y2]  (좌상, 우하)
    """
    if pts is None or len(pts) == 0:
        return None
    xs, ys = pts[:, 0], pts[:, 1]
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


# ---------------------------------------------------------------------------
# 2) 인접/포함 BBox 병합
# ---------------------------------------------------------------------------
def merge_close_bboxes(rects,
                       win_size=(512, 512),
                       margin=10,
                       iou_th=0.10):
    """
    rects   : [(x1,y1,x2,y2), ...]  from to_rect()
    win_size: (w, h) ─ 병합을 시도할 그리드 크기
    margin  : 두 박스 간 허용 gap (px)  ─ 클수록 많이 합쳐짐
    iou_th  : 확장된 박스 간 IoU ≥ iou_th 이면 같은 클러스터로 간주
    return  : 병합된 rect list
    """
    if not rects:
        return []

    rects = np.array(rects, dtype=int)
    w_win, h_win = win_size

    # ── ① spatial grid 로 coarse clustering
    grid = {}
    for idx, (x1, y1, x2, y2) in enumerate(rects):
        gx, gy = int(x1 // w_win), int(y1 // h_win)
        grid.setdefault((gx, gy), []).append(idx)

    merged, visited = [], np.zeros(len(rects), dtype=bool)

    # ── ② grid 안에서 DFS 로 이웃 탐색 + union
    for bucket in grid.values():
        for i in bucket:
            if visited[i]:
                continue
            stack, cluster = [i], []
            while stack:
                cur = stack.pop()
                if visited[cur]:
                    continue
                visited[cur] = True
                cluster.append(cur)
                for j in bucket:
                    if visited[j]:
                        continue
                    if _is_neighbor(rects[cur], rects[j], margin, iou_th):
                        stack.append(j)

            xs = rects[cluster, 0]
            ys = rects[cluster, 1]
            xe = rects[cluster, 2]
            ye = rects[cluster, 3]
            merged.append([int(xs.min()), int(ys.min()),
                           int(xe.max()), int(ye.max())])

    return merged


def _is_neighbor(a, b, margin, iou_th):
    """ margin 만큼 확장 후 IoU ≥ iou_th 또는 gap≤margin 이면 True """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ax1e, ay1e, ax2e, ay2e = ax1 - margin, ay1 - margin, ax2 + margin, ay2 + margin
    bx1e, by1e, bx2e, by2e = bx1 - margin, by1 - margin, bx2 + margin, by2 + margin

    inter_w = max(0, min(ax2e, bx2e) - max(ax1e, bx1e))
    inter_h = max(0, min(ay2e, by2e) - max(ay1e, by1e))
    if inter_w == 0 or inter_h == 0:
        return False

    inter = inter_w * inter_h
    area_a = (ax2e - ax1e) * (ay2e - ay1e)
    area_b = (bx2e - bx1e) * (by2e - by1e)
    iou = inter / (area_a + area_b - inter + 1e-6)
    return iou >= iou_th


# ---------------------------------------------------------------------------
# 3) 결과 저장 (txt + 시각화 jpg)
# ---------------------------------------------------------------------------
def saveBBOXResult(img_file, img_bgr, rects,
                   out_dir='./result_bbox/',
                   color=(0, 0, 255)):
    """
    img_file : 원본 이미지 경로 (이름 추출용)
    img_bgr  : np.ndarray (BGR)
    rects    : [(x1,y1,x2,y2), ...]
    """
    if rects is None:
        return
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(img_file))[0]

    # txt
    with open(os.path.join(out_dir, f'res_{name}.txt'), 'w') as f:
        for x1, y1, x2, y2 in rects:
            f.write(f'{x1},{y1},{x2},{y2}\n')

    # vis
    vis = img_bgr.copy()
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
    cv2.imwrite(os.path.join(out_dir, f'res_{name}.jpg'), vis)
# -----------------------------------------------------------
# 2-단계 말풍선 병합
# -----------------------------------------------------------
def merge_balloon(rects,
                  line_margin=15,      # 1차 병합용
                  line_iou=0.1,
                  y_overlap=0.5,       # 2차: 세로 오버랩 비율
                  x_gap=40):           # 2차: 가로 gap 허용치
    """
    rects : [(x1,y1,x2,y2), ...] word-level 직사각형
    return : 말풍선 단위 rect list
    """

    # --- 1차 : 기존 margin+IoU ---
    from itertools import combinations
    tmp = merge_close_bboxes(rects,
                             win_size=(512,512),
                             margin=line_margin,
                             iou_th=line_iou)

    tmp = np.array(tmp, int)
    used = np.zeros(len(tmp), bool)
    merged = []

    # --- 2차 : y-overlap & x-gap ---
    for i in range(len(tmp)):
        if used[i]: continue
        x1,y1,x2,y2 = tmp[i]
        group = [i]
        for j in range(i+1, len(tmp)):
            if used[j]: continue
            a = tmp[i]; b = tmp[j]
            # 세로 오버랩
            inter_h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
            min_h   = min(a[3]-a[1], b[3]-b[1])
            if inter_h / (min_h+1e-6) < y_overlap:
                continue
            # 가로 gap
            gap = max(0, max(b[0]-a[2], a[0]-b[2]))
            if gap > x_gap:
                continue
            group.append(j); used[j]=True

        xs = tmp[group,0]; ys = tmp[group,1]
        xe = tmp[group,2]; ye = tmp[group,3]
        merged.append([xs.min(), ys.min(), xe.max(), ye.max()])

    return merged

import numpy as np
from sklearn.cluster import DBSCAN

def merge_text_blocks(rects, dx_th=1.2, dy_th=1.0):
    rects = np.array(rects, float)
    xc = (rects[:,0]+rects[:,2])/2; yc = (rects[:,1]+rects[:,3])/2
    w  = rects[:,2]-rects[:,0] ;   h  = rects[:,3]-rects[:,1]

    def near(a,b):
        dx = abs(xc[a]-xc[b]) / ((w[a]+w[b])/2+1e-3)
        dy = abs(yc[a]-yc[b]) / ((h[a]+h[b])/2+1e-3)
        return (dx<=dx_th) and (dy<=dy_th)

    N=len(rects); used=[False]*N; blocks=[]
    for i in range(N):
        if used[i]: continue
        st=[i]; grp=[]
        while st:
            v=st.pop(); used[v]=True; grp.append(v)
            for u in range(N):
                if not used[u] and near(v,u): st.append(u)
        xs,ys=rects[grp,0],rects[grp,1]; xe,ye=rects[grp,2],rects[grp,3]
        blocks.append([int(xs.min()),int(ys.min()),int(xe.max()),int(ye.max())])
    return blocks