# class Bitboard:
#     def __init__(self):
#         self.current_position = 0
#         self.mask = 0
#         self.moves = 0  # 이동 횟수 추적

#     def can_play(self, col):
#         # 해당 열에 수를 둘 수 있는지 확인
#         return (self.mask & self.top_mask(col)) == 0

#     def play(self, col):
#         # 열에 수를 두고 현재 위치와 마스크 업데이트
#         self.current_position ^= self.mask
#         self.mask |= self.mask + self.bottom_mask(col)
#         self.moves += 1

#     @staticmethod
#     def top_mask(col):
#         # 열의 상단 비트 위치 반환
#         return 1 << ((height + 1) * col + height)

#     @staticmethod
#     def bottom_mask(col):
#         # 열의 하단 비트 위치 반환
#         return 1 << ((height + 1) * col)

#     def alignment(self):
#         # 정렬 확인
#         # 여기서는 수평, 대각선, 수직 정렬 확인 로직 구현
#         pass
