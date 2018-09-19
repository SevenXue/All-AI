# 200
class Solution:
    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """

        def area(grid, r, c):
            if (r < 0 or c < 0 or r >= len(grid) or c >= len(grid[0]) or grid[r][c] != '1'):
                return
            grid[r][c] = '#'
            area(grid, r, c + 1)
            area(grid, r, c - 1)
            area(grid, r + 1, c)
            area(grid, r - 1, c)

        total = 0
        for r in range(len(grid)):
            for c in range(len(grid[0])):
                if grid[r][c] == '1':
                    total += 1
                    area(grid, r, c)
        return total
