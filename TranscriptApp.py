# Import all the necessary dependencies
from flask import Flask, request
from youtube_transcript_api import YouTubeTranscriptApi
from langdetect import detect
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file
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
        summary = extractive_summarization(transcript)
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


def extractive_summarization(transcript):
    """
    Summarizes the input transcript using the Google Gemini API.
    The API analyzes the transcript and identifies the most salient sentences.

    Parameters:
    - transcript (str): The transcript text to be summarized.

    Returns:
    - summary (str): The summarized text.
    """
    genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
    model = genai.GenerativeModel('gemini-1.0-pro-latest')
    try:
        prompt_template = f"""
                            Task: Summarize the main points of the given YouTube video transcript in a concise manner.

                            Video Transcript:
                            {transcript}

                            Summary Instructions:
                            - Identify the key topics, ideas, and takeaways from the video transcript
                            - Organize the main points in a logical and coherent structure
                            - Maintain the original context and meaning of the video content
                            - Use clear and straightforward language in the summary

                            Summary:
                            """

        response = model.generate_content(prompt_template)

    except Exception as e:
        raise Exception(f"Error occurred during summarization: {str(e)}") from e

    formatted_response=f"Summary:\n\n{response.text}"

    return formatted_response


if __name__ == '__main__':
    application.run(debug=True)