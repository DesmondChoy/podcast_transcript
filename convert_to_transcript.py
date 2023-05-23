# %%
import whisper

# additional dependencies like ffmeg may need to be installed
# instructions: https://github.com/openai/whisper#setup

# %%
# https://github.com/openai/whisper#available-models-and-languages
# model = whisper.load_model("medium") #74m
model = whisper.load_model("base.en")  # 10m

# audio downloaded here: https://www.spreaker.com/user/manga-sensei/sol-rashidi-with-music
result = model.transcribe("sol_rashidi_with_disclaimer.mp3", verbose=True)
print(result["text"])
# %%
# Open a file in write mode ('w')
file = open("transcript_base_timestamps.txt", "w")

# Write the text to the file
file.write(result["text"])

# Close the file
file.close()
# %% [markdown]
# How different are the transcripts produced from medium and base.en model?
# %%
import nltk
from difflib import SequenceMatcher

nltk.download("punkt")  # for sentence tokenization


def read_and_tokenize_file(filename):
    with open(filename, "r") as file:
        text = file.read()
    return nltk.tokenize.sent_tokenize(text)


# Read and tokenize the transcripts
transcript_base = read_and_tokenize_file("transcript_base.txt")
transcript_med = read_and_tokenize_file("transcript_medium.txt")

# Initialize the matcher
matcher = SequenceMatcher(None, transcript_base, transcript_med)

# %% Check for matching blocks
for block in matcher.get_opcodes():
    if block[0] != "equal":
        print(
            f"Difference found in sentences starting at position {block[1]} in transcript_base and {block[3]} in transcript_med."
        )

        # print differing sentences
        for i in range(block[1], block[2]):
            print("transcript_base:", transcript_base[i])
        for i in range(block[3], block[4]):
            print("transcript_med:", transcript_med[i])
        print("\n")

# the differences are very close. Some samples:

# Difference found in sentences starting at position 38 in transcript_base and 45 in transcript_med.
# transcript_base: As a sophomore, you're in college.
# transcript_base: I was like, OK, maybe professional sports isn't in my, my teacher directory.
# transcript_med: As a sophomore year in college, I was like, OK, maybe professional sports isn't in my my trajectory.

# Difference found in sentences starting at position 112 in transcript_base and 125 in transcript_med.
# transcript_base: Everyone was just starting off and me and my leadership team at the time, we were one of the first teams to actually build it.
# transcript_med: Everyone was just starting off and me and my leadership team at the time.
# transcript_med: We were one of the first teams to actually build it.

# Difference found in sentences starting at position 180 in transcript_base and 196 in transcript_med.
# transcript_base: Doing all of that stuff for the fundamentals and that's defensive play.
# transcript_base: But what's going to gain and then I want to use a word note of variety, but what's going to help the business pay attention and keep their focus and have you continue to be a seat at the table is when you put when you develop your offensive playbook and that means you're sitting at the table with them and understanding what are their drivers?
# transcript_med: Doing all of that stuff are the fundamentals and that's defensive play.
# transcript_med: But what's going to gain and I don't want to use the word notoriety, but what's going to help the business pay attention and keep their focus and have you continue to be a seat at the table is when you play when you develop your offensive playbook.
# transcript_med: And that means you're sitting at the table with them and understanding what are their drivers?

# Difference found in sentences starting at position 262 in transcript_base and 285 in transcript_med.
# transcript_base: The funny thing about AI is it's a kitchen sink term right now.
# transcript_med: The funny thing about AI is, is it's a kitchen sink term right now.

# Difference found in sentences starting at position 572 in transcript_base and 589 in transcript_med.
# transcript_base: I just had the discipline.
# transcript_base: And I couldn't make up for it with natural smarts and charisma, so I just studied a lot.
# transcript_base: And I wasn't a natural test taker, so I just studied even more.
# transcript_med: I just had the discipline and I would, I couldn't make up for it with natural smarts and charisma.
# transcript_med: So I just studied a lot and I wasn't a natural test taker.
# transcript_med: So I just studied even more.
