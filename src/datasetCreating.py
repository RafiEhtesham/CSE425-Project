import lyricsgenius
import os
from dotenv import load_dotenv
import pandas as pd


def fetch_lyrics():
    load_dotenv()
    genius_api_key = os.getenv('ClientAccessToken')

    GTZAN_TITLE = "../data/GTZAN/song_title.csv"

    genius = lyricsgenius.Genius(genius_api_key, skip_non_songs=True, verbose=False)

    gtzan_title = pd.read_csv(GTZAN_TITLE)

    lyrics_data = gtzan_title.copy()
    lyrics_data['lyrics'] = None

    for idx, row in gtzan_title.iterrows():
        try:
            song = genius.search_song(title=row['songTitle'], artist=row['artistName'])
            if song:
                lyrics_data.at[idx, 'lyrics'] = song.lyrics
                print(f"✓ Found lyrics for {row['songTitle']} by {row['artistName']}")
            else:
                print(f"✗ No lyrics found for {row['songTitle']} by {row['artistName']}")
        except Exception as e:
            print(f"✗ Error fetching {row['songTitle']}: {str(e)}")

    # Save the dataset
    output_path = "../data/GTZAN/gtzan_with_lyrics.csv"
    lyrics_data.to_csv(output_path, index=False)
    print(f"\n✓ Dataset saved to {output_path}")

if __name__ == "__main__":
    fetch_lyrics()