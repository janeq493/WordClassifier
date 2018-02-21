import pygame
import matplotlib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.backends.backend_agg as agg
import pylab
fig = pylab.figure(figsize=[4, 4], dpi=100)
ax = fig.gca()
from NeuralNetwork import neuralNetwork
from data import data
BISQUE = (255, 228, 196)
WHEAT = (245, 212, 169)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

pygame.init()

scr = pygame.display.set_mode((1152, 576))
font = pygame.font.SysFont("calibri", 18)


learningRate = 0.2
batchSize = 25

instrList = ["1-train continously",
             "2-single training example", "3-user input", ""]

hist = []
lastCorrect = 0

itteration = 0
currentWord = ""
predicition = ""
confidence = ""
correctAnswer = ""
layerInfo = [26 * 18, 18, 5]
nn = neuralNetwork(26 * 18)
nn.addLayer(35, activationFunc='tanh')
nn.addLayer(5, costFunction="mse")

trainingData = data()

state = 0

categ = [
    "Spanish",
    "English",
    "Swahili",
    "Polish",
    "Japanese",
]
graph = []
loss = []
def write(w):
    #[224+i*int(640/len(layerInfo)),20+n*32]
    word = [font.render(ch, 1, BLACK) for ch in w]
    for i, let in enumerate(word):
        scr.blit(let, (219, 12 + i * 32))


def evaluate(preds, label):
    global hist,lastCorrect,confidence,predicition,correctAnswer,graph
    loss.append(nn.getCost(batchSize))
    highest = 0
    ans = 0
    correct = 0
    for i, p in enumerate(preds):
        if p > highest:
            highest = p
            ans = i
    confidence = str(highest)
    predicition = categ[ans]

    highest = 0
    for i, l in enumerate(label):
        if l > highest:
            highest = l
            correct = i
    correctAnswer = categ[correct]
    if len(hist) == 100:
        v = hist[0]
        hist = hist[1:]

        if v:
            lastCorrect -= 1
    if correct == ans:
        lastCorrect += 1
        hist.append(True)
        graph.append(lastCorrect/100)
        return True
    hist.append(False)
    graph.append(lastCorrect/100)
    return False

def updateGraph(graphdata):
    ax.plot(range(len(graphdata)),graphdata,'black',linewidth=.5)
    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.tostring_rgb()
    ax.set_autoscaley_on(False)
    pylab.ylim([0.0,1.0])

    graphSize = canvas.get_width_height()
    surf = pygame.image.fromstring(raw_data, graphSize, "RGB")
    scr.blit(surf, (762,180))



while True:
    # input
    for event in pygame.event.get():

        if event.type == pygame.KEYDOWN:
            state = 0
            if event.key == pygame.K_1 and state != 1:
                state = 1
            if event.key == pygame.K_2 and state != 2:
                state = 2
            if event.key == pygame.K_3 and state != 3:
                state = 3
            if event.key == pygame.K_4:
                #updateGraph(loss)
                #fig.savefig('graphloss.png', bbox_inches='tight')
                updateGraph(graph)
                fig.savefig('graphperf.png', bbox_inches='tight')
    # logic
    if (state == 1):
        train, labels, currentWord = trainingData.get(batchSize)
        preds = nn.feedForward(train[0])
        evaluate(preds, labels[0])
        nn.train(train, labels, learningRate, batchSize)
        itteration += 1
    if (state == 2):
        train, labels, currentWord = trainingData.get(1)
        preds = nn.feedForward(train[0])
        evaluate(preds, labels[0])
        nn.train(train, labels, learningRate, 1)

        state = 0
        itteration += 1
    if state == 3:
        state = 0

    if (itteration==300):
        updateGraph(graph)
        fig.savefig('graphperf300.png', bbox_inches='tight')
    if (itteration==20000):
        updateGraph(graph)
        fig.savefig('graphperf1200.png', bbox_inches='tight')



    # Draw

    # Background color
    pygame.draw.rect(scr, BISQUE, [0, 0, 1152, 576])
    pygame.draw.rect(scr, WHEAT, [192, 0, 512 + 128, 576])

    for i, l in enumerate(layerInfo):
        if i == 0:
            l = l // 26
        for n in range(l):
            pygame.draw.circle(
                scr, WHITE, [224 + i * int(640 / len(layerInfo)), 20 + n * 32], 13)

    # set up text
    instr = [font.render(t, 1, BLACK) for t in instrList]
    rightTxt = [
        font.render("Itteration: %d" % itteration, 1, BLACK),
        font.render("Current Word: " + currentWord, 1, BLACK),
        font.render("Learning Rate: " + str(learningRate), 1, BLACK),
        font.render("Batch size: %d" % batchSize, 1, BLACK),
    ]
    print(predicition)
    leftTxt = [
        font.render("Predicition: " + predicition, 1, BLACK),
        font.render("Confidence: " + confidence, 1, BLACK),
        font.render("Correct Answer: " + correctAnswer, 1, BLACK),
        font.render(str(lastCorrect) + "/%d = "%100 +
                    str(lastCorrect) + "%", 1, BLACK),
    ]

    write(currentWord)
    if state != 1:
        updateGraph(graph)


    for i in range(4):
        scr.blit(instr[i], (8, 8 + (18 * i)))
        scr.blit(rightTxt[i], (8, 128 + (i * 18) +
                               (int(i / 2) * (512 - 128 - 18))))
        scr.blit(leftTxt[i], (848, 8 + (i * 18) + (int(i / 1.5) * 40)))

    pygame.display.flip()
