import os

import hf_olmo
import torch

import json

import cv2
import webvtt

import whisper


from transformers import AutoModelForCausalLM, AutoTokenizer

from openai import OpenAI

from yt_dlp import YoutubeDL

DATABASE = "database"

EXAMPLE_COOKING_VIDOES = [
    "https://www.youtube.com/shorts/-OE6LhxsTNA", # short
    "https://www.youtube.com/shorts/V-eHFm7BQOY", # short
    "https://youtu.be/mhDJNfV7hjk", # long
    "https://youtu.be/vkcHmpKxFwg", # long
]

INVESTIGATION_VIDEOS = [
    #"https://www.youtube.com/watch?v=D_noTrlaYfk", # no verbal
    #"https://www.youtube.com/watch?v=YNxXsgINDvw", # no verbal
    #"https://www.youtube.com/watch?v=_0-o5Cw4mjQ", # no verbal
    "https://www.youtube.com/watch?v=aEFvNsBDCWs", # has verbal
    "https://www.youtube.com/shorts/fWp5z_YM07Q", # short
    "https://www.youtube.com/watch?v=cZ2KJPGVwNU", # has verbal
    "https://www.youtube.com/watch?v=gN-orgrgvU8", # has verbal
    "https://www.youtube.com/shorts/B-XGIGS4Ipw", # short
    #"https://www.youtube.com/watch?v=Q-w2c6PHKN0", # no verbal
    # old
    #"https://www.youtube.com/watch?v=AVx0oamyxLQ",
    #"https://youtu.be/Uuz8lWHOlh0?si=7gX38yinFFogxKbD",
    #"https://youtu.be/lllvs-H5-Q0?si=et1SWRdEmoL79IFc",
    #"https://www.youtube.com/watch?v=3Ho9ivMWhho",
    #"https://www.youtube.com/watch?v=O1CNMqvqEWg",
    #"https://youtu.be/aP0FzpzD6nU?si=a6qKkj1IploPIteJ",
]


olmo = AutoModelForCausalLM.from_pretrained("allenai/OLMo-7B", torch_dtype=torch.float16)
olmo = olmo.to('cuda')
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-7B")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print(OPENAI_API_KEY)
client = OpenAI(
    api_key=OPENAI_API_KEY,
)

video_library = {}

def download_video(video_link):
    # Download video 480p or, if short, whatever is available
    options = {
        'format': 'bv[height<=?480][ext=mp4]+ba[ext=mp3]/best',
		#'format': 'bestvideo[height<=480][ext=mp4]+bestaudio[ext=mp3]/best',
        'outtmpl': os.path.join(DATABASE, '%(id)s.%(ext)s'),
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': {'en'},  # Download English subtitles
        'subtitlesformat': '/vtt/g',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'keepvideo': True,
        'skip_download': False,
    }

    with YoutubeDL(options) as ydl:
        info = ydl.extract_info(video_link, download=False)
        metadata = ydl.sanitize_info(info)
        video_title = metadata.get('id')
        video_path = os.path.join(DATABASE, f'{video_title}.mp4')
        if not os.path.exists(video_path):
            ydl.download([video_link])
            print(f"Video '{video_title}' downloaded successfully.")
        else:
            print(f"Video '{video_title}' already exists in the directory.")
        return metadata

def extract_frames(video_path):
    video_cap = cv2.VideoCapture(video_path)
    
    frames = []
    
    while (True):
        res, frame = video_cap.read()
        if (res == False):
            break

        frames.append(frame)
    
    video_cap.release()

    return frames


def extract_transcript_from_audio(audio_path):
    output_path = audio_path.replace(".mp3", ".alt.json")
    raw_transcript = {}
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            raw_transcript = json.load(f)
    else:
        model = whisper.load_model("small.en")
        raw_transcript = model.transcribe(audio_path)
        with open(output_path, 'w') as f:
            json.dump(raw_transcript, f, indent=2)

    transcript = []
    for segment in raw_transcript["segments"]:
        transcript.append({
            "start": segment["start"],
            "finish": segment["end"],
            "text": segment["text"],
        })
    return transcript

def extract_transcript(subtitles_path, audio_path):
    if not os.path.exists(subtitles_path):
        print(f"Subtitles file '{subtitles_path}' does not exist.")
        if not os.path.exists(audio_path):
            print(f"Audio file '{audio_path}' does not exist.")
            return []
        transcript = extract_transcript_from_audio(audio_path)

        return transcript

    subtitles = webvtt.read(subtitles_path)

    transcript = []
    for caption in subtitles:
        lines = caption.text.strip("\n ").split("\n")
        if len(transcript) == 0:
            transcript.append({
                "start": caption.start,
                "finish": caption.end,
                "text": "\n".join(lines),
            })
            continue
        last_caption = transcript[len(transcript) - 1]

        new_text = ""
        for line in lines:
            if line.startswith(last_caption["text"], 0):
                new_line = line[len(last_caption["text"]):-1].strip()
                if len(new_line) > 0:
                    new_text += new_line + "\n"
            elif len(line) > 0:
                new_text += line + "\n"
        new_text = new_text.strip("\n ")
        if len(new_text) == 0:
            transcript[len(transcript) - 1]["finish"] = caption.end
        else:
            transcript.append({
                "start": caption.start,
                "finish": caption.end,
                "text": new_text,
            })
    return transcript

def process_video(video_link):
    metadata = download_video(video_link)
    
    video_title = metadata.get('id')
    print(f"'{video_title}'")

    video_library[video_title] = metadata

    video_path = os.path.join(DATABASE, f'{video_title}.mp4')
    subtitles_path = os.path.join(DATABASE, f'{video_title}.en.vtt')
    audio_path = os.path.join(DATABASE, f'{video_title}.mp3')

    video_frames = extract_frames(video_path)
    subtitles = extract_transcript(subtitles_path, audio_path)

    return video_frames, subtitles

def tokenize_video(video_link):
    video_frames, subtitles = process_video(video_link)

    return {
        "subtitles": subtitles,
    }

def run_olmo(messages):
    messages = [message["content"] for message in messages]

    inputs = tokenizer(messages, return_tensors='pt', return_token_type_ids=False).to('cuda')
    response = olmo.generate(**inputs, max_new_tokens=300, do_sample=True, top_p=0.6)
    return tokenizer.batch_decode(response, skip_special_tokens=True)[0]

def run_openai(messages):
    
    response = client.chat.completions.create(
        #model="gpt-3.5-turbo",
        model="gpt-4-0125-preview",
        messages=messages,
    )

    return [{
        "role": response.choices[0].message.role,
        "content": response.choices[0].message.content,
    }]

def run_llm(messages):
    return run_openai(messages)
    


def test_1():
    messages = [
        "Language model is",
        "Tilekbay Bekzat is",
        "KIXLAB is",
    ]    
    for message in messages:
        print(f"Input: {message}")
        response = run_llm(message)
        print(f"Output: {response}")

def annotate_messages_generate(result):
    system_message = """
    You are an assistant that describes the steps in a cooking video and outlines potential ambiguities, complexity, and specificity in the instructions.
    ----------------------------------------
    """
    user_message = ""
    for subtitle in result["subtitles"]:
        user_message += f"""
        {subtitle["start"]}-{subtitle["finish"]}: {subtitle["text"]}
        """
    user_message += """
    ----------------------------------------
    Given the above transcript, please provide the outline of each step, potential ambiguities, complexity, and specificity.
    Definitions:
    - Step: A distinct, outcome-oriented task within a procedural sequence, characterized by its purposeful contribution to achieving the overarching goal of the process. It encapsulates a set of actions or instructions that, while variable in execution, uniformly aim towards a specific result or transformation.
    - Description: A concise, informative account of the step's purpose, actions, and expected outcomes.
    - Ambiguities: Instances of uncertainty, vagueness, or inconsistency within the step that may hinder comprehension, execution, or evaluation with respect to the instructions provided in the video.
    - Complexity: The level of intricacy, difficulty, or sophistication associated with the step, encompassing the cognitive, physical, or temporal demands it imposes on the user.
    - Specificity: The degree of detail, precision, or clarity in the step's instructions, encompassing the extent to which the actions, tools, ingredients, or conditions are explicitly defined or elaborated upon.
    - Domain-specificity: The relevance, applicability, or pertinence of the step to different contexts, scenarios, or user groups, reflecting the extent to which the instructions are tailored to a particular domain, skill level, or cultural background.
    ----------------------------------------
    Follow the format below:
    Step 1: [Concise step label]
    Description: [Description of the step]
    Ambiguities: [Description of ambiguities]
    Complexity: [Description of complexity]
    Specificity: [Description of specificity]
    Domain-specificity: [Description of domain-specificity]
    ----------------------------------------
    """

    return [
        {
            "role": "system",
            "content": system_message,
        },
        {
            "role": "user",
            "content": user_message,
        },
    ]

def annotate_messages_reflect(messages, response):
    messages = messages + response
    user_message = """
    ----------------------------------------
    Please reflect on the generated steps, considering the definitions provided below.
    
    Definitions:
    - Step: A distinct, outcome-oriented task within a procedural sequence, characterized by its purposeful contribution to achieving the overarching goal of the process. It encapsulates a set of actions or instructions that, while variable in execution, uniformly aim towards a specific result or transformation.
    - Ambiguities: Instances of uncertainty, vagueness, or inconsistency within the step that may hinder comprehension, execution, or evaluation with respect to the instructions provided in the video.
    - Complexity: The level of intricacy, difficulty, or sophistication associated with the step, encompassing the cognitive, physical, or temporal demands it imposes on the user.
    - Specificity: The degree of detail, precision, or clarity in the step's instructions, encompassing the extent to which the actions, tools, ingredients, or conditions are explicitly defined or elaborated upon.
    - Domain-specificity: The relevance, applicability, or pertinence of the step to different contexts, scenarios, or user groups, reflecting the extent to which the instructions are tailored to a particular domain, skill level, or cultural background.
    
    Identify any discrepancies, inconsistencies, or areas of improvement in the generated annotation. Consider the following questions:
    1) Is the step clearly defined and distinct from other steps in the sequence?
    2) Is the description follow the definiton of a step? Does it clearly outline the purpose, actions, and expected outcomes?
    3) Is the ambigiuities section comprehensive? Does it address all potential sources of uncertainty or vagueness?
    4) Is the complexity section accurate? Does it reflect the level of difficulty or sophistication associated with the step?
    5) Is the specificity section detailed? Does it provide sufficient clarity on the actions, tools, ingredients, or conditions involved?
    6) Is the domain-specificity section relevant? Does it consider the context, scenario, or user group to which the step is tailored?

    Based on your reflections, suggest any modifications, additions, or clarifications to enhance the quality and accuracy of the annotation. 
    ----------------------------------------
    """
    return messages + [
        {
            "role": "user",
            "content": user_message,
        },
    ]

def annotate_messages_incorporate(messages, response):
    messages = messages + response
    user_message = """
    ----------------------------------------
    Based on the reflections and suggestions provided, revise the outline of the steps. Ensure that the revised annotation is clear, accurate, and comprehensive, reflecting the optimal structure and content for analyzing the cooking video.
    ----------------------------------------
    Follow the format below:
    Step 1: [Concise step label]
    Description: [Description of the step]
    Ambiguities: [Description of ambiguities]
    Complexity: [Description of complexity]
    Specificity: [Description of specificity]
    Domain-specificity: [Description of domain-specificity]
    ----------------------------------------
    """
    return messages + [
        {
            "role": "user",
            "content": user_message,
        },
    ]


def annotate_videos():

    for video_link in INVESTIGATION_VIDEOS:
        result = tokenize_video(video_link)
        print(json.dumps(result, indent=4))

        ## generate steps
        messages = annotate_messages_generate(result)
        for message in messages:
            print(f"{message['role']}: {message['content']}")
        response = run_llm(messages)
        for message in response:
            print(f"{message['role']}: {message['content']}")

        ## reflect on the steps
        messages = annotate_messages_reflect(messages, response)
        response = run_llm(messages)
        for message in response:
            print(f"{message['role']}: {message['content']}")

        ## incorporate feedback
        messages = annotate_messages_incorporate(messages, response)
        response = run_llm(messages)
        for message in response:
            print(f"{message['role']}: {message['content']}")


def classify_messages_generate(result):
    system_message = """
    You are an assistant that recognizes the cooking steps in the video and classifies the narrated content into given classes.
    ----------------------------------------
    """
    user_message = ""
    for subtitle in result["subtitles"]:
        user_message += f"""
        {subtitle["start"]}-{subtitle["finish"]}: {subtitle["text"]}
        """

    spectrum_basic = """
    - tacit: personal stories, context around the technique demonstrated in the step, etc.;
    - detailed: detailed explanation of the step and how it is performed;
    - basic: basic information about the step, what is happening;
    - minimal: minimal information about the step, just enough to understand what is happening;
    """

    spectrum_extended = """
    - tacit:
        -- Description: Narration that weaves personal anecdotes, contextual insights, and expert intuitions into the explanation of the technique. It doesn't just tell the viewer how to perform a step but enriches this guidance with why it's done this way, sharing the narrator's own experiences, mistakes, and lessons learned. This level might include comparative insights (e.g., what happens when you do it differently), metaphorical explanations that link the technique to more familiar experiences, and personal preference discussions that highlight the flexibility of the method.
        -- Example: "When folding the dough, I use a technique passed down from my grandmother, which, though seemingly gentle, integrates the ingredients thoroughly. Notice how my hands move—not just to combine but to feel the dough's texture, adjusting based on its response, much like tending a garden."
    - detailed:
        --Description: This goes beyond merely detailing the steps by incorporating explanations of underlying principles, such as the science behind a reaction, the technical rationale for a specific sequence, or the aesthetic reasons for a design choice. The narration aims to build a solid understanding, empowering viewers with the knowledge to adapt the technique to their contexts.
        --Example: "As I mix these ingredients, I'm creating what's known as an emulsion. It's a delicate balance—too fast, and it breaks, too slow, and it never forms. I'll guide you through finding that perfect rhythm, much like conducting an orchestra, where every instrument has its moment."
    -basic:
        --Description: Elevates the basic step description by briefly touching on the purpose or importance of each step within the larger procedure. While still concise, the explanation helps viewers grasp not just the 'what' but a hint of the 'why,' making each action more meaningful.
        --Example: "Now we're adding salt, which does more than just season. It strengthens the gluten structure, giving our bread its chewy texture. Think of it as the backbone of our loaf, subtly crucial."
    minimal:
        --Description: Provides the essential information needed to understand and follow the step, but with added clarity to prevent common misunderstandings. This level strives for concise effectiveness, ensuring that even with minimal detail, the viewer has a clear path forward.
        --Example: "We'll let the dough rest now—it's not just waiting; it's letting the gluten relax, making our next steps easier. It's like a short nap for the dough, preparing it for what's next."
    """


    user_message += f"""
    ----------------------------------------
    Given the above transcript, recognize all the cooking steps and classify the narrated content.

    Let's do this step-by-step:
    
    1. Recgnize a cooking step defined as: a distinct, outcome-oriented task within a procedural sequence, characterized by its purposeful contribution to achieving the overarching goal of the process. It encapsulates a set of actions or instructions that, while variable in execution, uniformly aim towards a specific result or transformation.
    
    2. For the recognized cooking step, find the relevant parts of the transcript, explain why is it relevant, and give a classification & reasoning for the class, based on the following classes:
    {spectrum_extended}

    
    Make sure that all the cooking steps are recognized and classified correctly.
    ----------------------------------------
    Follow the format below:
    Step 1: [Concise step label]
    Description: [Description of the step]
    Transcripts: [All transcripts related to the step]
    Explanation: [Explanation of the relevance of the transcripts]
    Classification: [Classification of the step]
    Reasoning: [Reasoning for the classification[]]
    ----------------------------------------
    """

    return [
        {
            "role": "system",
            "content": system_message,
        },
        {
            "role": "user",
            "content": user_message,
        },
    ]

def classify_videos():
    for video_link in INVESTIGATION_VIDEOS[0:5]:
        result = tokenize_video(video_link)
        combined_subtitles = ""
        for subtitle in result["subtitles"]:
            combined_subtitles += f"{subtitle['text']}\n"
        combined_subtitles = combined_subtitles.replace("  ", " ").replace("\n\n", "\n").replace("\n", " ")
        print(combined_subtitles)
        continue

        ## generate steps
        messages = classify_messages_generate(result)
        for message in messages:
            print(f"{message['role']}: {message['content']}")
        response = run_llm(messages)
        for message in response:
            print(f"{message['role']}: {message['content']}")

        # ## reflect on the steps
        # messages = classify_messages_reflect(messages, response)
        # response = run_llm(messages)
        # for message in response:
        #     print(f"{message['role']}: {message['content']}")

        # ## incorporate feedback
        # messages = classify_messages_incorporate(messages, response)
        # response = run_llm(messages)
        # for message in response:
        #     print(f"{message['role']}: {message['content']}")

def main():
    # annotate_videos()
    classify_videos()

if __name__ == "__main__":
    main()
