"""
Voice Recording Script for Voice Biometric Authentication Project
Run this on your laptop to collect your genuine voice samples.

Install dependencies first:
    pip install sounddevice soundfile numpy
"""

import sounddevice as sd
import soundfile as sf
import numpy as np
import os
import time
import sys

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
SPEAKER_NAME  = 'abhiram'       # your name
SAMPLE_RATE   = 16000           # Hz
CHANNELS      = 1               # mono
SAVE_DIR      = 'my_voice_data' # folder created next to this script

PASSPHRASES = {
    'phrase1': 'My voice is my password',
    'phrase2': 'Open sesame authenticate now',
    'phrase3': 'Verify my identity please',
}

FREE_SPEECH_PROMPTS = [
    'Count from one to twenty slowly',
    'Describe your voice biometric project',
    'The quick brown fox jumps over the lazy dog',
    'Say the days of the week and months of the year',
    'Speech recognition systems use deep neural networks',
    'Biometric authentication verifies human identity securely',
    'Count backwards from twenty to one',
    'Describe what you did today in a few sentences',
    'Deep learning has revolutionized speaker verification',
    'My name is Abhiram and I am a B.Tech student',
]
# ─────────────────────────────────────────────────────────────────────────────


def setup_dirs():
    dirs = [
        f'{SAVE_DIR}/text_dependent/phrase1',
        f'{SAVE_DIR}/text_dependent/phrase2',
        f'{SAVE_DIR}/text_dependent/phrase3',
        f'{SAVE_DIR}/text_independent',
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print(f'Save folder: {os.path.abspath(SAVE_DIR)}')


def count_existing(folder):
    return len([f for f in os.listdir(folder) if f.endswith('.wav')])


def record_clip(duration):
    print(f'  Recording... (speak now)', end='', flush=True)
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype='float32'
    )
    for i in range(duration, 0, -1):
        print(f' {i}s', end='', flush=True)
        time.sleep(1)
    sd.wait()
    print(' Done.')
    return audio.flatten()


def play_back(audio):
    print('  Playing back...')
    sd.play(audio, SAMPLE_RATE)
    sd.wait()


def save_wav(audio, filepath):
    sf.write(filepath, audio, SAMPLE_RATE)


def prompt_continue():
    while True:
        choice = input('  Keep this recording? [y/n/q to quit]: ').strip().lower()
        if choice in ('y', ''):
            return 'keep'
        elif choice == 'n':
            return 'redo'
        elif choice == 'q':
            return 'quit'


def record_session(folder, prefix, phrase_text, target_count, duration):
    existing = count_existing(folder)
    if existing >= target_count:
        print(f'  Already have {existing}/{target_count} recordings. Skipping.')
        return

    print(f'\n  Target: {target_count} recordings  |  Already done: {existing}')
    print(f'  Say: "{phrase_text}"')
    print(f'  Duration: {duration}s per recording')
    print()

    idx = existing + 1
    while idx <= target_count:
        print(f'--- Recording {idx}/{target_count} ---')
        input('  Press Enter when ready...')

        audio = record_clip(duration)
        play_back(audio)

        action = prompt_continue()
        if action == 'keep':
            filename = f'{folder}/{SPEAKER_NAME}_{prefix}_{idx:03d}.wav'
            save_wav(audio, filename)
            print(f'  Saved: {filename}')
            idx += 1
        elif action == 'redo':
            print('  Redoing this recording.')
        elif action == 'quit':
            print('  Session paused. Run again to continue.')
            sys.exit(0)


def show_menu():
    print('\n' + '=' * 55)
    print('  VOICE BIOMETRIC AUTHENTICATION — RECORDING TOOL')
    print('=' * 55)
    print(f'  Speaker : {SPEAKER_NAME}')
    print(f'  Save to : {os.path.abspath(SAVE_DIR)}')
    print()
    print('  [1] Record Phrase 1  — text-dependent (30 clips)')
    print('  [2] Record Phrase 2  — text-dependent (30 clips)')
    print('  [3] Record Phrase 3  — text-dependent (30 clips)')
    print('  [4] Record all 3 phrases in one session')
    print('  [5] Record free speech — text-independent (10 clips)')
    print('  [6] Show recording progress')
    print('  [0] Exit')
    print()
    return input('  Choose an option: ').strip()


def show_progress():
    print('\n--- Recording Progress ---')
    total_needed = 0
    total_done   = 0
    for key, phrase in PASSPHRASES.items():
        folder = f'{SAVE_DIR}/text_dependent/{key}'
        done   = count_existing(folder)
        needed = 30
        total_done   += done
        total_needed += needed
        bar = ('█' * done + '░' * (needed - done))[:30]
        print(f'  {key}: [{bar}] {done}/{needed}  "{phrase}"')
    folder = f'{SAVE_DIR}/text_independent'
    done   = count_existing(folder)
    needed = 10
    total_done   += done
    total_needed += needed
    bar = ('█' * done + '░' * (needed - done))[:10]
    print(f'  free  : [{bar}] {done}/{needed}  (free speech)')
    print(f'\n  Total : {total_done}/{total_needed} recordings complete')


def main():
    setup_dirs()

    while True:
        choice = show_menu()

        if choice == '1':
            print('\n=== Phrase 1 ===')
            record_session(
                folder       = f'{SAVE_DIR}/text_dependent/phrase1',
                prefix       = 'phrase1',
                phrase_text  = PASSPHRASES['phrase1'],
                target_count = 30,
                duration     = 4
            )

        elif choice == '2':
            print('\n=== Phrase 2 ===')
            record_session(
                folder       = f'{SAVE_DIR}/text_dependent/phrase2',
                prefix       = 'phrase2',
                phrase_text  = PASSPHRASES['phrase2'],
                target_count = 30,
                duration     = 4
            )

        elif choice == '3':
            print('\n=== Phrase 3 ===')
            record_session(
                folder       = f'{SAVE_DIR}/text_dependent/phrase3',
                prefix       = 'phrase3',
                phrase_text  = PASSPHRASES['phrase3'],
                target_count = 30,
                duration     = 4
            )

        elif choice == '4':
            print('\n=== All 3 Phrases ===')
            for key, phrase in PASSPHRASES.items():
                print(f'\n>>> Now recording: {key}')
                record_session(
                    folder       = f'{SAVE_DIR}/text_dependent/{key}',
                    prefix       = key,
                    phrase_text  = phrase,
                    target_count = 30,
                    duration     = 4
                )

        elif choice == '5':
            print('\n=== Free Speech (Text-Independent) ===')
            folder   = f'{SAVE_DIR}/text_independent'
            existing = count_existing(folder)
            print(f'  Existing: {existing}/{len(FREE_SPEECH_PROMPTS)}')

            for i, prompt in enumerate(FREE_SPEECH_PROMPTS, 1):
                filename = f'{folder}/{SPEAKER_NAME}_free_{i:03d}.wav'
                if os.path.exists(filename):
                    print(f'  Skipping {i} (already recorded)')
                    continue
                print(f'\n--- Free Speech {i}/{len(FREE_SPEECH_PROMPTS)} ---')
                print(f'  Say: "{prompt}"')
                while True:
                    input('  Press Enter when ready...')
                    audio  = record_clip(duration=8)
                    play_back(audio)
                    action = prompt_continue()
                    if action == 'keep':
                        save_wav(audio, filename)
                        print(f'  Saved: {filename}')
                        break
                    elif action == 'redo':
                        print('  Redoing.')
                    elif action == 'quit':
                        sys.exit(0)

        elif choice == '6':
            show_progress()

        elif choice == '0':
            print('\nExiting. Run again to continue recording.')
            break

        else:
            print('Invalid option. Try again.')


if __name__ == '__main__':
    main()
