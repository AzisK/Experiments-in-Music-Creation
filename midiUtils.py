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

def getHand(hand):
    if hand == 'r':
        return 'Piano right'
    elif hand == 'l':
        return 'Piano left'
    else:
        print('ERROR. No hand selected: Select either "r" for "Right hand or "l" for "Left hand"')
        return 'No hand selected'

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

def readNotes(piece, time=0):
  # Array of tuples to create a dataframe later
  data = []
  # Array of all piano notes to hold the times of their activation
  on = np.zeros(128, dtype=int)
  # Read the file
  midi = mido.MidiFile(piece)

  # Get the tracks and iterate over them
  for track in midi.tracks:
    # Iterate over the messages in the track
    for msg in track:
      # Add the time of messages to the overall time
      time = time + msg.time
      # Check if it is a note message
      if msg.type == 'note_on':
        # Check if was pressed 
        if msg.velocity > 0:
          # If it was pressed, get the time of this action
          on[msg.note] = time
        # Check if it was released
        else:
          # Just a safety check to ensure it was pressed before
          if on[msg.note]:
            # Check if the notes belong to any hand
            if (isAnyHand(track.name)):
              # Get the hand number (0 - left, 1 - right)
              hand = getHandNumber(track.name)
              # Get the length of the note
              length = time - on[msg.note]
              # Does not include notes of less than the length of 4 ticks 
              # or the minimum hearing length
              if length > 4: 
                # Append a note
                data.append((msg.note, on[msg.note], time, length, hand))
              # Unpress the note
              on[msg.note] = 0

  # Return the dataframe
  return pd.DataFrame(data, columns=['pitch', 'on', 'off', 'length', 'hand'])

def stateToMessage(stateMatrix):
    data = []
    notes = np.zeros(128, dtype=int)
    for index, state in enumerate(stateMatrix):
        for i in range(128):
            if state[i] == 1:
                if notes[i] != 1:
                    notes[i] = 1
                    time = index * 60
                    data.append((147, i, 80, time))
            if index != 0:
                if state[i] == 0:
                    if stateMatrix[index - 1, i] == 1:
                        notes[i] = 0
                        time = (index - 1) * 60
                        data.append((147, i, 0, time))
    sortedData = sorted(data, key=lambda x: x[-1])
    return sortedData

def toMidi(tracks, filename='Midi.mid'):
    song = mido.MidiFile()
    for messages in tracks:
        track = mido.MidiTrack()
        song.tracks.append(track)
        time = 0
        for message in messages:
            message = np.array(message)
            now = message[-1]
            delta = now - time
            time = now
            message[-1] = delta
            msg = mido.Message.from_bytes(message[:3])
            msg.time = delta
            track.append(msg)
    song.save(filename)