## 백준 2468번
import sys
from collections import deque

input = sys.stdin.readline
sys.setrecursionlimit(10 ** 6)
queue = deque()

def bfs(a, b):
    dx = [1, -1, 0, 0]
    dy = [0, 0, 1, -1]
    queue = deque([[a, b]])
    visited[a][b] = True

    while queue:
        x, y = queue.popleft()

        # 상/하/좌/우 탐색
        for i in range(4):
            nx = x + dx[i]
            ny = y + dy[i]

            # 범위 내에 있고 탐색하지 않았다면
            if 0 <= nx < n and 0 <= ny < m and not visited[nx][ny]:
                # 빙산이라면 큐에 추가
                if graph[nx][ny] != 0:
                    visited[nx][ny] = True
                    queue.append([nx, ny])

                # 빙산이 아니라면 바닷물로 카운트
                else:
                    cnt[x][y] += 1
    return 1


n, m = map(int, sys.stdin.readline().split())
graph = [list(map(int, sys.stdin.readline().split())) for _ in range(n)]
year = 0

# 모든 얼음이 녹을 때까지 반복한다.
while True:
    answer = []
    visited = [[False] * m for _ in range(n)]
    cnt = [[0] * m for _ in range(n)]

    # 빙산과 접촉되어 있는 바닷물 확인
    for i in range(n):
        for j in range(m):
            if graph[i][j] != 0 and not visited[i][j]:
                answer.append(bfs(i, j))

    # 빙산을 깍아줌
    for i in range(n):
        for j in range(m):
            graph[i][j] -= cnt[i][j]
            if graph[i][j] < 0:
                graph[i][j] = 0

    if len(answer) == 0 or len(answer) >= 2:
        break

    year += 1

print(year) if len(answer) >= 2 else print(0)