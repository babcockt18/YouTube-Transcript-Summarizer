# Import all the necessary dependencies
from flask import Flask, request
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.tokenize import sent_tokenize
from langdetect import detect
from google.cloud import language_v1

application = Flask(__name__)

@application.get('/summary')
def summary_api():
    """
    Summarizes the transcript of a YouTube video.

    This function takes a YouTube video URL and an optional max_length parameter as inputs.
    It first retrieves the transcript of the YouTube video.
    If the transcript is longer than 3000 words, it uses extractive summarization (e.g. LSA).
    Otherwise, it uses abstractive summarization.

    Parameters:
    - url (str): The URL of the YouTube video.
    - max_length (int, optional): The maximum length of the summary. Defaults to 150.

    Returns:
    - str: The summarized transcript.
    - int: HTTP status code (200 for success, 404 for failure).
    """
    url = request.args.get('url', '')
    max_length = int(request.args.get('max_length', 150))
    video_id = url.split('=')[1]

    try:
        transcript = get_transcript(video_id)
    except:
        return "No subtitles available for this video", 404

    try:
        # Extractive summarization using Google Gemini API
        if len(transcript.split()) > 3000:
            summary = extractive_summarization(transcript)
        else:
            summary = abstractive_summarization(transcript, max_length)
    except Exception as e:
        print(f"Error occurred during summarization: {str(e)}")
        return "An error occurred during summarization. Please try again later.", 500

    return summary, 200

def is_transcript_english(transcript):
    """
    Detect if the transcript is primarily in English.

    :param transcript: The transcript text to be analyzed.
    :return: True if the transcript is primarily in English, False otherwise.
    """
    try:
        language = detect(transcript)
        return language == 'en'

    except Exception as e:
        return False


def get_transcript(video_id):
    """
    Fetches and concatenates the transcript of a YouTube video.

    Parameters:
    video_id (str): The ID of the YouTube video.

    Returns:
    str: A string containing the concatenated transcript of the video.

    Raises:
    Exception: If there is an error in fetching the transcript.
    """
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
    except Exception as e:
        raise e

    transcript = ' '.join([d['text'] for d in transcript_list])
    return transcript

def abstractive_summarization(transcript, max_length):
    """
    Summarizes the given transcript using an abstractive summarization model.

    The function employs an NLP pipeline for summarization and applies it to chunks
    of the input transcript. The chunks are processed independently and concatenated
    to form the final summary.

    Parameters:
    - transcript (str): The transcript text to be summarized.
    - max_length (int): The maximum length of the summary. It controls how concise
                       the summary should be.

    Returns:
    - summary (str): The summarized text.
    """
    summarizer = pipeline('summarization')
    summary = ''
    for i in range(0, (len(transcript)//1000) + 1):
        summary_text = summarizer(transcript[i * 1000:(i+1) * 1000], max_length=max_length)[0]['summary_text']
        summary = summary + summary_text + ' '
    return summary

def extractive_summarization(transcript):
    """
    Summarizes the input transcript using the Google Gemini API.
    The API analyzes the transcript and identifies the most salient sentences.

    Parameters:
    - transcript (str): The transcript text to be summarized.

    Returns:
    - summary (str): The summarized text.
    """
    client = language_v1.LanguageServiceClient()

    document = language_v1.Document(content=transcript, type_=language_v1.Document.Type.PLAIN_TEXT)
    response = client.analyze_entities(document=document, encoding_type='UTF32')

    # Get the salience scores for each sentence
    sentences = response.sentences
    salience_scores = [sentence.sentiment.score for sentence in sentences]

    # Rank sentences based on salience scores
    ranked_sentences = [item[0] for item in sorted(enumerate(salience_scores), key=lambda item: -item[1])]

    # Select top sentences for summary
    num_sentences = int(0.4 * len(sentences))  # 20% of the original sentences
    selected_sentences = sorted(ranked_sentences[:num_sentences])

    # Compile the final summary
    summary = " ".join([sentences[idx].text.content for idx in selected_sentences])
    return summary


if __name__ == '__main__':
    application.run(debug=True)

# TODO: Add translation ui menu and translation functionality