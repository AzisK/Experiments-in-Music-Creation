# -*- coding: utf-8 -*-
import mido
import numpy as np
from pathlib import Path
import pandas as pd

def getPieces():
  # Set the path
  path = Path('data')
  # Return all midis in the path
  return list(path.glob('*.mid'))

def readPieces():
  # Array of tuples to create a dataframe later
  frames = []
  # Initiate time
  time = 0
  # Get pieces and iterate over them
  for piece in getPieces():
    # Get the dataframe and length of the piece 
    df, time = readNotes(piece, time)
    # Append the dataframe to frames
    frames.append(df)

  # Concatenate all the datarames into one
  dfAll = pd.concat(frames)
  dfAll.to_csv('music.csv', index=False)

def isAnyHandOrPedal(trackName):
  if trackName == 'Piano right' or trackName == 'Piano left' or trackName == 'Pedal':
    return True
  else:
    return False

def getHandNumber(trackName):
  if trackName == 'Piano right':
    return 1
  elif trackName == 'Piano left' or trackName == 'Pedal':
    return 0
  else:
    print('ERROR. No hand returned: return 0 for "Piano left" or "Pedal" and 1 for "Piano right"')

def loadPieces(force=False):
  if force:
    readNotes(readPieces)
  return pd.read_csv('music.csv', index_col=False)

def readNotes(piece, time=0):
  # Array of tuples to create a dataframe later
  data = []
  # Array of all piano notes to hold the times of their activation for the left hand
  onLeft = np.zeros(128, dtype=int)
  # Array of all piano notes to hold the times of their activation for the right hand
  onRight = np.zeros(128, dtype=int)
  # Read the file
  midi = mido.MidiFile(piece)
  # Time will have to be reordered for different hand notes
  notesStarted = False
  # Initiate right and left hand side times
  rightMaxTime = 0
  leftMaxTime = 0
  # Initiate piece length
  pieceLength = 0
  # Set start time
  startTime = time

  # Get the tracks and iterate over them
  for track in midi.tracks:
    # Check if the notes belong to any hand
    if (isAnyHandOrPedal(track.name)):
      # Iterate over the messages in the track
      for msg in track:
        # Add the time of messages to the overall time
        # Check if notes have started
        if notesStarted: 
          time = time + msg.time
        # Get the hand number (0 - left, 1 - right)
        hand = getHandNumber(track.name)
        # Check if it is a note message
        if msg.type == 'note_on':
          # Check if notes have started
          if not notesStarted: 
            # Start the time from beginning in this case
            time = startTime + msg.time
            # Set that notes started
            notesStarted = True
          # Check if it was pressed 
          if msg.velocity > 0:
            # Workaround to get truthy value for a note that was pressed at time 0
            if time == 0:
              time = -1
            # Check which hand (0 - left, 1 - right)
            if hand:
              # If it was pressed, get the time of this action
              onRight[msg.note] = time
            else:
              onLeft[msg.note] = time
            # Bring time back to zero if workaround
            if time == -1:
              time = 0
          # Check if it was released
          elif msg.velocity == 0:
            # Initiate start
            start = 0
            if hand:
              # Just a safety check to ensure it was pressed before
              if not onRight[msg.note]:
                continue
              # Set start 
              start = onRight[msg.note]
              # Unpress the note
              onRight[msg.note] = 0
            else:
              if not onLeft[msg.note]:
                continue
              start = onLeft[msg.note]
              onLeft[msg.note] = 0

            # If workaround, get start to 0
            if start == -1:
              start = 0

            # Update the piece length
            if time > pieceLength:
                pieceLength = time

            # Get the length of the note
            length = time - start
            # Append a note
            data.append((msg.note, start, time, length, hand))

      # After the end of the track, set notesStarted to false        
      notesStarted = False

  return pd.DataFrame(data, columns=['pitch', 'on', 'off', 'length', 'hand']), pieceLength

def toStateMatrix(df, quant=60):
  length = df['off'].values[-1]
  steps = int(length / quant)
  stateMatrix = np.zeros((steps, 128), dtype=int)
  for row in df.itertuples():
      pitch = row[1]
      on = row[2]
      off = row[3]
      stateMatrix[int((on / quant)) : int((off / quant + 1)), pitch] = 1
  return stateMatrix

def state2Tuples(stateMatrix):
  data = []
  notes = np.zeros(128, dtype=int)
  for index, state in enumerate(stateMatrix):
      for i in range(128):
        if index != 0:
          if state[i] == 0:
              if stateMatrix[index - 1, i] == 1:
                  notes[i] = 0
                  time = index - 1
                  data.append((147, i, 0, time))
        if state[i] == 1:
          if notes[i] != 1:
              notes[i] = 1
              time = index
              data.append((147, i, 70, time))
          
  sortedData = sorted(data, key=lambda x: x[-1])
  return sortedData

def tuples2Midi(messages, filename='Midi.mid'):
    song = mido.MidiFile()
    track = mido.MidiTrack()
    song.tracks.append(track)
    time = 0
    for message in messages:
        message = np.asarray(message)
        now = message[-1]
        delta = now - time
        time = now
        message[-1] = delta
        msg = mido.Message.from_bytes(message[:3])
        msg.time = delta
        track.append(msg)
    song.save(filename)

def quantize(tick, quant=60):
    res = tick % quant
    if res != 0:
        if res < (quant / 2):
            tick -= res
        if res >= (quant / 2):
            tick += (quant - res)
    return tick

def findLength(on, off):
    return off - on

def findLengthForDf(df):
    df['length'] = df.apply(lambda x: findLength(x['on'], x['off']), axis=1)

def quantizeDf(df, quant):
    df['on'] = df['on'].apply(lambda x: quantize(x, quant))
    df['off'] = df['off'].apply(lambda x: quantize(x, quant))
    findLengthForDf(df)