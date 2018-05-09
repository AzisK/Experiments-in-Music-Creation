import matplotlib.pyplot as plt
import midiUtils as mu
from pathlib import Path
import pandas as pd
import numpy as np

def generateCSV(force=False):
  my_file = Path('music.csv')
  if not my_file.is_file() or force: 
    mu.readPieces()

df = mu.loadPieces()

def getNotesHist():
  plt.hist(df['pitch'], bins=128)
  plt.xlabel('Pitch')
  plt.ylabel('Times')
  plt.show()

def getNotesRighHist():
  mydf = df[df['hand'] == 1]
  plt.hist(mydf['pitch'], bins=128)
  plt.show()

def getNotesLeftHist():
  mydf = df[df['hand'] == 0]
  plt.hist(mydf['pitch'], bins=128)
  plt.show()

def getLengthHist():
  plt.hist(df['length'], bins=200)
  plt.xlabel('Length')
  plt.ylabel('Times')
  plt.show()

def getLengthQuantHist():
  mu.quantizeDf(df)
  plt.hist(df['length'], bins=200)
  plt.xlabel('Length')
  plt.ylabel('Times')
  plt.show()

def getLengthRightHist():
  mydf = df[df['hand'] == 1]
  plt.hist(mydf['length'], bins=200)
  plt.show()

def getLengthLeftHist():
  mydf = df[df['hand'] == 0]
  plt.hist(mydf['length'], bins=200)
  plt.show()

def getNotesBounds():
  minp = df['pitch'].min()
  maxp = df['pitch'].max()
  print(minp, maxp)

def getStats():
  df = mu.loadPieces()
  mu.quantizeDf(df)
  data = mu.toStateMatrix(df, 29, 91)

  mean = data.mean()
  coldiff = data - np.array([data.mean(1)]).transpose()
  std = np.mean(np.sqrt(np.square(coldiff)))
  print("MEAN: {0}, STD: {1}".format(mean, std))

# getNotesHist()
# getNotesRighHist()
# getNotesLeftHist()

# getLengthHist()
# getLengthRightHist()
# getLengthLeftHist()

# getLengthQuantHist()
# getNotesBounds()

getStats()