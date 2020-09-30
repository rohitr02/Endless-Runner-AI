import neat
from neat import nn
import pygame
import os
import random
import math
import neatlib.visualize as visualize
import pickle
import matplotlib.pyplot as plt
import numpy as np

pygame.init()
WIN_Y = 250
WIN_X = 500
Obj_Dim = 50
STAT_FONT = pygame.font.SysFont("comicsans", 50)
gen = 0
start = False

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


class EnemyBlock:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.velX = random.uniform(Min_Enemy_VelX, 2*Min_Enemy_VelX)

    def move(self):
        self.x -= self.velX
        # if self.x <= Obj_Dim*-1:
        #     self.velX = random.uniform(Min_Enemy_VelX, 2 * Min_Enemy_VelX)
        #     self.x = WIN_X + Obj_Dim

    def draw(self, win):
        pygame.draw.rect(win, Enemy_Color, (self.x, self.y, Obj_Dim, Obj_Dim))


class MainBlock:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.velY = 0.0
        self.jumping = False

    def jump(self):
        if self.y == Block_StartY or self.jumping:
            self.jumping = True
            self.y -= abs(Jump_Vel)
            if self.y <= Obj_Dim:
                self.jumping = False

    def move(self):
        if self.jumping:
            self.jump()
        elif self.y < Block_StartY:
            self.y -= Jump_Vel

    def draw(self, win):
        pygame.draw.rect(win, Block_Color, (self.x, self.y, Obj_Dim, Obj_Dim))


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
    global score
    score = 0
    clock = pygame.time.Clock()
    running = True
    global start
    while running:
        # clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    start = True
        if start:
            if score == 200 and len(blocks) <= 1:
                running = False
            if len(blocks) == 0:
                score = 0
                running = False
            else:
                for x, block in enumerate(blocks):
                    if check_collision(block, enemy):
                        if len(blocks) == 1:
                            if gen == 0 or gen % 100 == 0:
                                visualize.draw_net(config, ge[x], False, str(gen) + " net")
                        if score < 1:
                            ge[x].fitness -= 10
                        else:
                            ge[x].fitness -= 9/score
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
                        block.move()
                        if block.y == Block_StartY:
                            ge[x].fitness += math.sqrt(score+1)*0.3
                        # else:
                            # ge[x].fitness -= 0.01
                        output = nets[x].activate((block.x, block.y, enemy.x, enemy.velX*-1))

                        if output[0] > 0.5:
                            block.jump()
                            # ge[x].fitness -= 1/(score+0.5)

                    draw_window(win, blocks, enemy, score, gen)
    gen += 1


def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(main, 1000)
    with open("bestmodel.pickle", "wb") as f:
            pickle.dump(winner, f)
    visualize.draw_net(config, winner, False, "winner net")
    visualize.plot_species(stats)
    visualize.plot_stats(stats)


    ## load existing model

    # pickle_in = open("bestmodel.pickle", "rb")
    # model = pickle.load(pickle_in)
    # # visualize.draw_net(config, model, False, "saved")
    #
    # attempts = list(range(1, 101))
    # scores = []
    # for num in attempts:
    #     scores.append(run_with_model(nn.FeedForwardNetwork.create(model, config)))
    # mean_scores = [np.mean(scores)] * len(attempts)
    # fig, ax = plt.subplots()
    # ax.plot(attempts, scores, label='Data', marker='o')
    # ax.plot(attempts, mean_scores, label='Avg', marker='_')
    # plt.xlabel("Attempt number")
    # plt.ylabel("Score")
    # ax.legend(loc='upper right')
    # plt.title("Best Net Scores")
    #
    # plt.savefig("avg scores.png")


def run_with_model(model):
    # win = pygame.display.set_mode((WIN_X, WIN_Y))
    block = MainBlock(Block_StartX, Block_StartY)
    enemy = EnemyBlock(Enemy_StartX, Enemy_StartY)
    score = 0
    clock = pygame.time.Clock()
    running = True
    while running:
        # clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                quit()
        # if score == 100:
        #     running = False
        #     return score
        if check_collision(block, enemy):
            running = False
        else:
            enemy.move()

            if enemy.x <= Obj_Dim * -1:
                enemy.velX = random.uniform(Min_Enemy_VelX, 2 * Min_Enemy_VelX)
                enemy.x = WIN_X + Obj_Dim
                score += 1

            block.move()

            output = model.activate((block.x, block.y, enemy.x, enemy.velX))

            if output[0] > 0.5:
                block.jump()

            # win.fill(pygame.Color("black"))
            # block.draw(win)
            # enemy.draw(win)
            # text = STAT_FONT.render("Score: " + str(score), 1, (255, 255, 255))
            # win.blit(text, (WIN_X - 10 - text.get_width(), 10))
            # pygame.display.update()
    return score


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "conf-feedforward.txt")
    run(config_path)
