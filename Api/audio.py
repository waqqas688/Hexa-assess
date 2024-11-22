import whisper
import pyaudio
import wave
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity1
from textblob import TextBlob

def record_audio(output_filename="input_audio.wav", duration=10)
    print("Recording... Speak now!")
    chunk = 1024
    format = pyaudio.paInt16
    channels = 1
    rate = 44100

    p = pyaudio.PyAudio()
    stream = p.open(format=format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)
    frames = []

    for _ in range(0, int(rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(output_filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(format))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()
    print("Recording finished.")

def transcribe_audio(audio_file):
    print("Transcribing your answer...")
    model = whisper.load_model("small")
    result = model.transcribe(audio_file, language='en')
    print(f"Transcribed text: {result['text']}")
    return result['text']

def evaluate_response(candidate_answer, question, keywords):
    print("Evaluating your response...")
    try:
        # Semantic similarity
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform([candidate_answer, question])
        semantic_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

        # Sentiment analysis
        sentiment_score = TextBlob(candidate_answer).sentiment.polarity

        # Keyword matching
        candidate_tokens = word_tokenize(candidate_answer.lower())
        matched_keywords = set(candidate_tokens).intersection(set(keywords))

        keyword_score = len(matched_keywords) / len(keywords) if keywords else 0

        # Aggregate score
        average_score = (semantic_similarity + sentiment_score + keyword_score) / 3

        print(f"Semantic Similarity Score: {round(semantic_similarity, 2)}")
        print(f"Sentiment Score: {round(sentiment_score, 2)}")
        print(f"Keyword Score: {round(keyword_score, 2)}")
        print(f"Overall Score: {round(average_score * 100, 2)}%")
    except Exception as e:
        print(f"Error during evaluation: {e}")

def main():
    while True:
        print("\nOptions:")
        print("1. Start Listening")
        print("2. Exit")
        choice = input("Enter your choice: ")

        if choice == "1":
            question = "What are your skills?"
            keywords = ["skills", "programming", "development"]
            print(f"\nQuestion: {question}")
            print(f"Expected Keywords: {keywords}")

            record_audio()
            candidate_answer = transcribe_audio("input_audio.wav")
            evaluate_response(candidate_answer, question, keywords)
        elif choice == "2":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
