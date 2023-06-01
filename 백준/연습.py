## 백준 2468번
import sys
from collections import deque

input = sys.stdin.readline
sys.setrecursionlimit(10 ** 6)
queue = deque()

n, m = map(int, input().split())
graph = [list(map(int, input().split())) for _ in range(n)]
visited = [[0 for _ in range(m)] for _ in range(n)]
dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]
def dfs(x, y):
    if x <= -1 or x > n or y <= -1 or y > m:
        return False
    if visited[x][y] == 0:
        visited[x][y] = 1
        dfs(x-1, y)
        dfs(x+1, y)
        dfs(x, y-1)
        dfs(x, y+1)
        return True
    return False

def ice_melting(x, y):
    queue.append(x, y)
    while queue:
        x, y = queue.popleft()
        for i in range(4):
            nx = x + dx[i]
            ny = y + dy[i]
            if 0 <= nx < m and 0 <= ny < n and graph[nx][ny] == 0:
                graph[x][y] -= 1
                queue.append((nx, ny))
            if graph[x][y] == 0:
                queue.clear()
                break
            
cnt = 0
max_year = max(map(max, graph))

for i in range(max_year):
    for j in range(n):
        for k in range(m):
            if dfs(j, k) == True:
                cnt += 1
                