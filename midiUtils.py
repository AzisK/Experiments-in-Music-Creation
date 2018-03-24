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
  # Get pieces and iterate over them
  for piece in getPieces():
    # Check if frames is not empty
    if frames:
      # Get the time of the end of last note 
      time = frames[-1]['off'].values[-1]
      # Get the dataframe (send starting time as well) of the piece and append it to frames
      frames.append(readNotes(piece, time))
    else:
      # Get the dataframe and append it to frames
      frames.append(readNotes(piece))

  # Concatenate all the datarames into one
  df = pd.concat(frames)
  df.to_csv('music.csv', index=False)

def isAnyHand(trackName):
  if trackName == 'Piano right' or trackName == 'Piano left':
    return True
  else:
    return False

def getHandNumber(trackName):
  if trackName == 'Piano right':
    return 1
  elif trackName == 'Piano left':
    return 0
  else:
    print('ERROR. No hand returned: return 0 for "Piano left" and 1 for "Piano right"'

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
  # Do not start adding the notes until they started
  notesStarted = False
  # Time will have to be reordered for left hand notes
  leftNotesStarted = False

  # Get the tracks and iterate over them
  for track in midi.tracks:
    # Iterate over the messages in the track
    for msg in track:
      # Check if the notes belong to any hand
      if (isAnyHand(track.name)):
        # Check if the notes started
        if notesStarted:
          # Add the time of messages to the overall time
          time = time + msg.time
        # Check if it is a note message
        if msg.type == 'note_on':
          # Set notesStarted to true
          notesStarted = True
          # Get the hand number (0 - left, 1 - right)
          hand = getHandNumber(track.name)
          # Check if it was pressed 
          if msg.velocity > 0:
            # Check which hand (0 - left, 1 - right)
            if hand:
              # If it was pressed, get the time of this action
              onRight[msg.note] = time
            else:
              # Check if left hand side notes have started
              if not leftNotesStarted: 
                # Start the time from beginning in this case
                time = msg.time
                # Set that the left hand side notes started
                leftNotesStarted = True
              onLeft[msg.note] = time
          # Check if it was released
          elif msg.velocity == 0:
            start = 0
            if hand:
              # Just a safety check to ensure it was pressed before
              if onRight[msg.note]:
                start = onRight[msg.note]
                # Unpress the note
                onRight[msg.note] = 0
            else:
              if onLeft[msg.note]:
                start = onLeft[msg.note]
                onLeft[msg.note] = 0

            # Get the length of the note
            length = time - start
            # Does not include notes of less than the length of 4 ticks 
            # or the minimum hearing length
            if length > 4: 
              # Append a note
              data.append((msg.note, start, time, length, hand))

  return pd.DataFrame(data, columns=['pitch', 'on', 'off', 'length', 'hand'])

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