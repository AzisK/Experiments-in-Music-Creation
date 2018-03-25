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

def loadPieces():
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
           # Check which hand (0 - left, 1 - right)
            if hand:
              # If it was pressed, get the time of this action
              onRight[msg.note] = time
            else:
              onLeft[msg.note] = time
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
              # If time is more than max, set max to this time
              if time > rightMaxTime:
                rightMaxTime = time
              # Unpress the note
              onRight[msg.note] = 0
            else:
              if not onLeft[msg.note]:
                continue
              start = onLeft[msg.note]
              if time > leftMaxTime:
                leftMaxTime = time
              onLeft[msg.note] = 0

            # Update the piece length
            if leftMaxTime > rightMaxTime and leftMaxTime > time:
              pieceLength = leftMaxTime
            elif rightMaxTime > leftMaxTime and rightMaxTime > time:
              pieceLength = rightMaxTime

            # Get the length of the note
            length = time - start
            # Does not include notes of less than the length of 4 ticks 
            # or the minimum hearing length
            if length > 4: 
              # Append a note
              data.append((msg.note, start, time, length, hand))

      # After the end of the track, set notesStarted to false        
      notesStarted = False

  return pd.DataFrame(data, columns=['pitch', 'on', 'off', 'length', 'hand']), pieceLength

def toStateMatrix(df):
  length = df['off'].values[-1]
  stateMatrix = np.zeros((length, 128), dtype=int)
  for row in df.itertuples():
      pitch = row[1]
      on = row[2]
      off = row[3]
      stateMatrix[on : off, pitch] = 1
  return stateMatrix

def stateToTuples(stateMatrix):
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

def tuplesToMidi(messages, filename='Midi.mid'):
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