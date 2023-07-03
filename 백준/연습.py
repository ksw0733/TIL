import sys
input = sys.stdin.readline
def solution(n, arr1, arr2):
    answer = []
    
    for i in range(n):
        tmp = bin(arr1[i] | arr2[i])
        tmp = tmp[2:]
        print(tmp)
        tmp = tmp.replace('1', '#').replace('0', ' ')
        answer.append(tmp)
    return answer

n = int(input())
arr1 = list(map(int, input().split()))
arr2 = list(map(int, input().split()))

solution(n, arr1, arr2)

print(bin(31)[2:], bin(14)[2:], sep='\n')