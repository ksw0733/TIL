## 백준 14503번
import sys
from collections import deque

input = sys.stdin.readline

def dfs(x, y, v):
    global res
    
    if graph[x][y] == 0:
        graph[x][y] = 2
        res += 1
        
    for _ in range(4):
        nv = (v + 3) % 4
        nx = x + dx[nv]
        ny = y + dy[nv]
        
        if graph[nx][ny] == 0:
            dfs(nx, ny, nv)
            return
        
        v = nv
    
    nv = (v + 2) % 4
    nx = x + dx[nv]
    ny = y + dy[nv]
    
    if graph[nx][ny] == 1:
        return
    
    dfs(nx, ny, v)
    

n, m = map(int, input().split())
r, c, d = map(int, input().split())
graph = [list(map(int, input().split())) for _ in range(n)]

dx = [-1, 0, 1, 0]
dy = [0, 1, 0, -1]
res = 0
dfs(r, c, d)
print(res)