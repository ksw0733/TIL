## 백준 2606번
import sys
from collections import deque

input = sys.stdin.readline

n = int(input())

x, y = map(int, input().split())

m = int(input())
graph = [[False] * (n + 1) for _ in range(n + 1)]
visited = [False] * (n + 1)
cnt = 0

def dfs(graph, x, y):
    global cnt
    cnt += 1
    visited[x] = True
    for i in range(m+1):
        
    
for _ in range(m):
    a, b = map(int, input().split())
    graph[a][b] = True
    graph[b][a] = True

dfs(graph, x, visited)

print(cnt)