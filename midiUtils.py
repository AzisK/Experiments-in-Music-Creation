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
def loadPieces(force=False) -> pd.DataFrame:
    if force:
        return readPieces()

    return pd.read_csv('music.csv')


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

def toStateMatrix(df, minp=0, maxp=127, quant=60):
  length = df['off'].values[-1]
  steps = int(length / quant)
  bounds = maxp + 1 - minp
  stateMatrix = np.zeros((bounds, steps), dtype=int)
  for row in df.itertuples():
      pitch = row[1] - minp
      on = row[2]
      off = row[3]
      hand = row[5]
      if hand == 1:
        stateMatrix[pitch, int(on / quant) : int(off / quant + 1)] = 1
      elif hand == 0:
        stateMatrix[pitch, int(on / quant) : int(off / quant + 1)] = -1
  return stateMatrix

def state2Tuples(stateMatrix, minp, quant):
  data = []
  heights = np.shape(stateMatrix)[0]
  notes = np.zeros(heights, dtype=int)

  lengths = np.shape(stateMatrix)[1]
  for step in range(lengths):
    for note, state in enumerate(stateMatrix[:, step]):
      if step != 0:
        if state == 0:
          if stateMatrix[note, step - 1]:
            notes[note] = 0
            time = (step - 1) * quant
            data.append((147, note + minp, 0, time))
      if state != 0:
        if notes[note] == 0:
          notes[note] = 1
          time = step * quant
          data.append((147, note + minp, 70, time))

  sortedData = sorted(data, key=lambda x: x[-1])
  return sortedData

def tuples2Midi(messages, filename='Midi.mid'):
  print('Initialising MIDI file {}'.format(filename))
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

def quantizeOn(tick, quant=60):
  res = tick % quant
  if res != 0:
    if res < (quant / 2):
      tick -= res
    if res >= (quant / 2):
      tick += (quant - res)
  return tick

def quantizeOff(tickOn, tick, quant=60):
  res = tick % quant
  if res != 0:
    if res < (quant / 2):
      tick -= res
    if res >= (quant / 2):
      tick += (quant - res)
  if tick == tickOn:
    tick += quant
  return tick

def findLength(on, off):
  return off - on

def quantizeDf(df, quant=60):
  df['on'] = df.apply(lambda x: quantizeOn(x['on'], quant), axis=1)
  df['off'] = df.apply(lambda x: quantizeOff(x['on'], x['off'], quant), axis=1)
  df['length'] = df.apply(lambda x: findLength(x['on'], x['off']), axis=1)
