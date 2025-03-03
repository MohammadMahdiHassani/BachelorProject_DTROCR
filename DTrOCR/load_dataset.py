import re
from pathlib import Path

# Define the Word class
class Word:
    def __init__(self, word_id, file_path, writer_id, transcription):
        self.word_id:str = word_id
        self.file_path:Path = file_path
        self.writer_id:str = writer_id
        self.transcription:str = transcription
    
    def __repr__(self):
        return (f"Word(id='{self.word_id}', file_path=PosixPath('{self.file_path}'), "
                f"writer_id='{self.writer_id}', transcription='{self.transcription}')")

# The input string (as an example)
input_string = "Word(id='a01-000u', file_path=PosixPath('./iam_words/words/a01/a01-000u/a01-000u-00-00.png'), writer_id='000', transcription='A') /"

# Regular expression to extract the components
pattern = r"Word\(id='([^']+)',\s*file_path=PosixPath\('([^']+)'\),\s*writer_id='([^']+)',\s*transcription='([^']+)'\)"

# Match the pattern
match = re.search(pattern, input_string)

# Check if the match was successful
if match:
    word_id = match.group(1)
    file_path = Path(match.group(2))  # Convert the string to a Path object
    writer_id = match.group(3)
    transcription = match.group(4)
    
    # Create the Word object
    word_object = Word(word_id, file_path, writer_id, transcription)
    
    # Show the resulting object
    print(word_object.file_path)
else:
    print("No match found.")
