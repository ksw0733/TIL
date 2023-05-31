## 백준 5014번
import sys
from collections import deque

input = sys.stdin.readline

f, s, g, u, d = map(int, input().split())

building = [0] * (f + 1)
queue = deque()

def bfs(s):
    queue.append(s)
    building[s] = 1
    while queue:
        x = queue.popleft()
        
        if x == g:
            return building[x] -1
        
        for move in (x + u, x - d):
            if 1 <= move <= f and building[move] == 0:
                building[move] = building[x] + 1
                queue.append(move)
                
    return 'use the stairs'

print(bfs(s))