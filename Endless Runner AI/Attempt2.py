import neat
from neat import nn
import pygame
import os
import random
import math
import neatlib.statistics as statistics
import neatlib.visualize as visualize
import pickle

pygame.init()
WIN_Y = 250
WIN_X = 500
Obj_Dim = 50
STAT_FONT = pygame.font.SysFont("comicsans", 50)
gen = 0

# Block vars
Block_Color = pygame.Color("blue")
Jump_Vel = -7
Jump_Height = 200
Block_StartX = 100
Block_StartY = WIN_Y - Obj_Dim

# Obj vars
Enemy_Color = pygame.Color("red")
Min_Enemy_VelX = abs(Jump_Vel/1.25)
Enemy_StartX = WIN_X - Obj_Dim
Enemy_StartY = WIN_Y - Obj_Dim


# Make it randomized: 50% chance of being top obj or btm obj. if top then y-pos is randomized between 1.5 * obj_dim and Jumpheight
# Otherwise make it regular btm

class EnemyBlock:
    def __init__(self, x, y):
        self.x = x
        if not self.is_top_or_btm():
            self.y = y
        else:
            self.y = random.uniform(1.5 * Obj_Dim, Jump_Height)
        self.velX = random.uniform(Min_Enemy_VelX, 2 * Min_Enemy_VelX)

    def move(self):
        self.x -= self.velX
        # if self.x <= Obj_Dim*-1:
        #     self.velX = random.uniform(Min_Enemy_VelX, 2 * Min_Enemy_VelX)
        #     self.x = WIN_X + Obj_Dim

    # True for top, false for btm
    def is_top_or_btm(self):
        if random.random() < 0.5:
            return True
        return False

    def draw(self, win):
        pygame.draw.rect(win, Enemy_Color, (self.x, self.y, Obj_Dim, Obj_Dim))


class MainBlock:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.height = Obj_Dim
        self.width = Obj_Dim
        self.velY = 0.0
        self.jumping = False

    def jump(self):
        if (self.y == Block_StartY or self.jumping) and self.width == Obj_Dim:
            self.jumping = True
            self.y -= abs(Jump_Vel)
            if self.y <= Obj_Dim:
                self.jumping = False

    def duck(self):
        if self.width == Obj_Dim and self.y == Block_StartY:
            self.width = Obj_Dim/2
            self.y += self.width

    def unduck(self):
        if self.width == Obj_Dim/2:
            self.y -= self.width
            self.width = Obj_Dim

    def move(self):
        if self.jumping:
            self.jump()
        elif self.y < Block_StartY:
            self.y -= Jump_Vel

    def draw(self, win):
        pygame.draw.rect(win, Block_Color, (self.x, self.y, self.height, self.width))


def check_collision(block, enemy):
    if (block.x - Obj_Dim < enemy.x < block.x + Obj_Dim) and (block.y - Obj_Dim < enemy.y < block.y + Obj_Dim):
        return True
    return False


def draw_window(win, blocks, enemy, score, gen):
    win.fill(pygame.Color("black"))
    for block in blocks:
        block.draw(win)
    enemy.draw(win)
    text = STAT_FONT.render("Score: " + str(score), 1, (255, 255, 255))
    win.blit(text, (10, 10))

    text = STAT_FONT.render("Gen: " + str(gen), 1, (255, 255, 255))
    win.blit(text, (WIN_X - 10 - text.get_width(), 10))

    text = STAT_FONT.render("Alive: " + str(len(blocks)), 1, (255, 255, 255))
    win.blit(text, (WIN_X - 10 - text.get_width(), 10 + text.get_height()))

    pygame.display.update()


def main(genomes, config):
    nets = []
    ge = []
    global gen
    blocks = []

    for _,g in genomes:
        net = nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        blocks.append(MainBlock(Block_StartX, Block_StartY))
        g.fitness = 0
        ge.append(g)

    win = pygame.display.set_mode((WIN_X, WIN_Y))
    # block = MainBlock(Block_StartX, Block_StartY)
    enemy = EnemyBlock(Enemy_StartX, Enemy_StartY)
    score = 0
    clock = pygame.time.Clock()
    running = True
    while running:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                quit()

        if len(blocks) == 0:
            score = 0
            running = False
        else:
            for x, block in enumerate(blocks):
                if check_collision(block, enemy):
                    blocks.pop(x)
                    nets.pop(x)
                    ge.pop(x)
            if len(blocks) == 0:
                running = False
            else:
                enemy.move()

                if enemy.x <= Obj_Dim * -1:
                    enemy.velX = random.uniform(Min_Enemy_VelX, 2 * Min_Enemy_VelX)
                    enemy.x = WIN_X + Obj_Dim
                    score += 1
                    for x, block in enumerate(blocks):
                        block.unduck()
                    if enemy.is_top_or_btm():
                        enemy.y = random.uniform(1.5 * Obj_Dim, Jump_Height)
                    else:
                        enemy.y = Enemy_StartY

                for x, block in enumerate(blocks):
                    block.move()
                    ge[x].fitness += math.sqrt(score)*0.1
                    output = nets[x].activate((block.x, block.y, enemy.x, enemy.y, enemy.velX))
                    if output[0] > 0.5 and output[1] < 0.5:
                        block.jump()

                    if output[0] < 0.5 and output[1] > 0.5:
                        block.duck()

                draw_window(win, blocks, enemy, score, gen)
    gen += 1


def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(main, 1000)
    with open("bestmodel2.pickle", "wb") as f:
            pickle.dump(winner, f)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "conf-feedforward2.txt")
    run(config_path)
