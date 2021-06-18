import pygame
import math
from queue import PriorityQueue

WIDTH = 800
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("A* - DFS - BFS - Dijkstra Path Finding Algorithm")

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
GREY = (128, 128, 128)
TURQUOISE = (64, 224, 208)


class Spot:
    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.width = width
        self.total_rows = total_rows
        self.x = row * width
        self.y = col * width
        self.color = WHITE
        self.neighbors = []
        self.visited = False

    def get_pos(self):
        return self.row, self.col

    def is_close(self):
        return self.color == RED

    def is_open(self):
        return self.color == GREEN

    def is_barrier(self):
        return self.color == BLACK

    def is_start(self):
        return self.color == ORANGE

    def is_end(self):
        return self.color == TURQUOISE

    def reset(self):
        self.color = WHITE

    def make_closed(self):
        self.color = RED

    def make_open(self):
        self.color = GREEN

    def make_start(self):
        self.color = ORANGE

    def make_end(self):
        self.color = TURQUOISE

    def make_path(self):
        self.color = PURPLE

    def make_barrier(self):
        self.color = BLACK

    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

    def update_neighbors(self, grid):
        self.neighbors = []
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier():  # DOWN
            self.neighbors.append(grid[self.row + 1][self.col])

        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier():  # UP
            self.neighbors.append(grid[self.row - 1][self.col])

        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier():  # RIGHT
            self.neighbors.append(grid[self.row][self.col + 1])

        if self.col > 0 and not grid[self.row][self.col - 1].is_barrier():  # LEFT
            self.neighbors.append(grid[self.row][self.col - 1])

    def get_neighbors(self):
        return self.neighbors

    def __lt__(self, other):
        return False

    def is_visited(self):
        return self.visited

    def visit(self):
        self.visited = True


def h(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)


def make_grid(rows, width):
    grid = []
    gap = width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            spot = Spot(i, j, gap, rows)
            grid[i].append(spot)
    return grid


def draw_grid(win, rows, width):
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap))
        for j in range(rows):
            pygame.draw.line(win, GREY, (j * gap, 0), (j * gap, width))


def draw(win, grid, rows, width):
    win.fill(WHITE)

    for row in grid:
        for spot in row:
            spot.draw(win)

    draw_grid(win, rows, width)
    pygame.display.update()


def reconstruct_path(came_from, current, draw):
    while current in came_from:
        current = came_from[current]
        current.make_path()
        draw()


def get_clicked_pos(pos, rows, width):
    gap = width // rows
    y, x = pos

    row = y // gap
    col = x // gap

    return row, col


def dfs_algorithm(draw, grid, start, end):
    print('running')
    first_start = start
    came_from = {}
    for spot in first_start.get_neighbors():
        came_from[spot] = first_start
        if spot != start:
            spot.make_closed()
        if spot == end:
            return True
        spot.make_open()
        if not spot.is_visited():
            spot.visit()
            draw()
            if dfs_algorithm(draw, grid, spot, end):
                reconstruct_path(came_from, spot, draw)
                return True
            else:
                continue
    return False


def bfs_algorithm(draw, grid, start, end):
    print('bfs')
    queue = [start]
    came_from = {}
    while len(queue) != 0:
        current = queue.pop(0)
        if current != start:
            current.make_closed()
        for spot in current.get_neighbors():
            spot.make_open()
            if not spot.is_visited():
                spot.visit()
                came_from[spot] = current
                queue.append(spot)
                draw()
            if spot == end:
                reconstruct_path(came_from, spot, draw)
                return True
    return False


def dijkstra_algorithm(draw, grid, start, end):
    count = 0
    open_set = PriorityQueue()
    came_from = {}
    open_set.put((0, start))
    while not open_set.empty():
        current = open_set.get()[1]
        if current == end:
            return True
        for spot in current.get_neighbors():
            if not spot.is_visited():
                spot.visit()
                spot.make_open()
                came_from[spot] = current
                open_set.put((h(current.get_pos(), end.get_pos()), spot))
                draw()
        if current != start:
            current.make_closed()
    return False


def algorithm(draw, grid, start, end):
    print('running al')
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}
    g_score = {spot: float("inf") for row in grid for spot in row}
    g_score[start] = 0
    f_score = {spot: float("inf") for row in grid for spot in row}
    f_score[start] = h(start.get_pos(), end.get_pos())

    open_set_hash = {start}

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = open_set.get()[2]
        open_set_hash.remove(current)

        if current == end:
            end.make_end()
            return True

        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1

            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + h(neighbor.get_pos(), end.get_pos())
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()

        draw()

        if current != start:
            current.make_closed()

    return False


def main(win, width):
    rows = 50
    grid = make_grid(rows, width)

    start = None
    end = None

    run = True
    started = False

    while run:
        draw(win, grid, rows, width)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if started:
                continue

            if pygame.mouse.get_pressed()[0]:  # LEFT CLICK
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, rows, width)
                spot = grid[row][col]
                if not start and spot != end:
                    start = spot
                    start.make_start()
                elif not end and spot != start:
                    end = spot
                    end.make_end()

                elif spot != end and spot != start:
                    spot.make_barrier()

                elif pygame.mouse.get_pressed()[2]:
                    pos = pygame.mouse.get_pos()
                    row, col = get_clicked_pos(pos, rows, width)
                    print(row, col)
                    spot = grid[row][col]
                    spot.reset()
                    if spot == start:
                        start = end
                    elif spot == end:
                        end = start
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a and start and end:
                    for row in grid:
                        for spot in row:
                            spot.update_neighbors(grid)

                    algorithm(lambda: draw(win, grid, rows, width), grid, start, end)

                # After setting up, press B to run BFS algorithm
                if event.key == pygame.K_b and start and end:
                    for row in grid:
                        for spot in row:
                            spot.update_neighbors(grid)

                    bfs_algorithm(lambda: draw(win, grid, rows, width), grid, start, end)
                # After setting up, press C to run Dijkstra algorithm
                if event.key == pygame.K_c and start and end:
                    for row in grid:
                        for spot in row:
                            spot.update_neighbors(grid)
                    dijkstra_algorithm(lambda: draw(win, grid, rows, width), grid, start, end)
                    
                # After setting up, press D to run DFS algorithm
                if event.key == pygame.K_d and start and end:
                    for row in grid:
                        for spot in row:
                            spot.update_neighbors(grid)
                    dfs_algorithm(lambda: draw(win, grid, rows, width), grid, start, end)

main(WIN, WIDTH)
