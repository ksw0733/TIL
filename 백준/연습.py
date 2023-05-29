## 백준 7569번
import sys
from collections import deque

input = sys.stdin.readline

m, n, h = map(int, input().split())

dx = [0, 0, -1, 1, 0, 0]
dy = [-1, 1, 0, 0, 0, 0]
dz = [0, 0, 0, 0, -1, 1]

visited = [False]*((n*m*h)+1)

def dfs(x, y, z):
    if  x<= -1 or x >= m or y <= -1 or y >= n or z <= -1 or z >= h:
        return False
    if graph[x][y][z] == 1:
        graph[x][y][z] = 1
        dfs(x-1, y, z)
        dfs(x+1, y, z)
        dfs(x, y-1, z)
        dfs(x, y+1, z)
        dfs(x, y, z-1)
        dfs(x, y, z+1)
        return True
    return False

graph = []
for i in range(h):
    for j in range(n):
        graph.append(list(map(int, input().split())))