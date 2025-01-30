import agentpy as ap
import json
import time
import numpy as np
import socket
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns, IPython
from matplotlib import pyplot as plt, cm
import heapq


# Define movement directions (Up, Down, Left, Right)
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

#  Heuristic function: Manhattan distance
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

#  A* Algorithm


class MazeAgent(ap.Agent):
    '''
    Initializing agent elements:
    - Has a route and starts with next step
    '''
    def setup(self):

        self.env = self.model.env
        self.route = self.env.a_star_search(self.env.environment, self.p.start, self.p.goal)
        self.next_step = 1

    '''
    Actual action execution. d
    '''
    def execute(self):
      next_pos = self.route[self.next_step]
      self.model.env.move_to(self, next_pos)
      self.next_step += 1
      pos = self.get_position()
      return {"x": pos[0], "y": 0, "z": pos[1]}
      
    def get_position(self):
        return self.env.positions[self]



class Maze(ap.Grid):
    def setup(self):
        # Initialize the maze environment
        self.environment = np.copy(self.p.maze)
        self.environment[self.p.goal] = self.p.goal_value

    '''
    A* Algorithm. Calculates the shortest path from start to goal.
    '''
    def a_star_search(self, grid, start, goal):
      rows, cols = grid.shape
      open_set = []
      heapq.heappush(open_set, (0, start))  # (cost, position)

      came_from = {}  # Track the path
      g_score = {start: 0}  # Cost from start to each node
      f_score = {start: heuristic(start, goal)}  # Estimated total cost

      while open_set:
          _, current = heapq.heappop(open_set)

          if current == goal:
              #  Reconstruct and return the path
              path = []
              while current in came_from:
                  path.append(current)
                  current = came_from[current]
              path.append(start)
              return path[::-1]  # Reverse the path

          for dx, dy in DIRECTIONS:
              neighbor = (current[0] + dx, current[1] + dy)

              #  Check boundaries
              if not (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols):
                  continue

              #  Check if the cell is impassable
              if grid[neighbor] in [-1, -10]:
                  continue

              #  Calculate new cost
              tentative_g_score = g_score[current] + grid[neighbor]
              if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                  came_from[neighbor] = current
                  g_score[neighbor] = tentative_g_score
                  f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                  heapq.heappush(open_set, (f_score[neighbor], neighbor))

      return None

'''
Maze model. definition of class MazeModel.
'''
class MazeModel(ap.Model):
    def setup(self):
        self.env = Maze(self, shape=maze.shape)
        self.agent = MazeAgent(self)
        self.env.add_agents([self.agent], positions=[self.p.start])
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind(("127.0.0.1", 65432))
        self.socket.listen(1)
        print("Waiting for Unity connection...")
        self.conn, self.addr = self.socket.accept()
        print(f"Connected to {self.addr}")


    def step(self):
        data = self.agent.execute()
        message = json.dumps(data)
        try:
            self.conn.sendall(message.encode("utf-8"))
        except BrokenPipeError:
            print("Connection lost.")
            self.stop()
        time.sleep(1)

    def update(self):
        if self.agent.get_position() == self.model.p.goal:
            print('ending')
            self.stop()

    # Report found route and Q-values
    def end(self):
        print("Closing connection...")
        self.conn.close()
        self.socket.close()
        print("Connection closed.")



explorer, goal = -3, -2

maze = np.load("streets-1.npy")

parameters = {
    'maze': maze,
    'start': (5,3),
    'goal': (16, 26),
    'goal_value': 10000,
    'steps': 150,
}


model = MazeModel(parameters)
model.run(steps=50)

# Environment representation with a grid

