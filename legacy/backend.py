import os

# import hf_olmo
import torch

import json

import cv2
import webvtt

import whisper

import re


# from transformers import AutoModelForCausalLM, AutoTokenizer

from openai import OpenAI

from yt_dlp import YoutubeDL

DATABASE = "static/database"

OUTPUT_FOLDER = "synthetic_data"

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
    "https://www.youtube.com/watch?v=gN-orgrgvU8", # has verbal
    "https://www.youtube.com/watch?v=cZ2KJPGVwNU", # has verbal
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

EMBEDDING_VIDEOS = {
    "11967": [
        "https://www.youtube.com/watch?v=yJQShkjNn08",
        "https://www.youtube.com/watch?v=yweUoYP1v_o",
        "https://www.youtube.com/watch?v=Ehntsffsx08",
        "https://www.youtube.com/watch?v=tdk9_Xs_CC0",
        "https://www.youtube.com/watch?v=dkhy4vn9HcY",
        "https://www.youtube.com/watch?v=QECo58lV-bE",
        "https://www.youtube.com/watch?v=SMh2sjuEwxM",
        "https://www.youtube.com/watch?v=DaEzhwLFPi8",
        "https://www.youtube.com/watch?v=J_5scvrv0LU",
        "https://www.youtube.com/watch?v=umbBEHlpTfo",
        "https://www.youtube.com/watch?v=pq_INi_4IBI",
        "https://www.youtube.com/watch?v=pYOQutHfCDo",
    ],
    "13630": [
        "https://www.youtube.com/watch?v=wwyCUOijLeE",
        "https://www.youtube.com/watch?v=WOQxVNAVwGk",
        "https://www.youtube.com/watch?v=zP57hPjfx2Y",
        "https://www.youtube.com/watch?v=KSIrBt6G5yQ",
        "https://www.youtube.com/watch?v=zNCo7PHUZr8",
        # "https://www.youtube.com/watch?v=0nPKzOHUIXw",
        # "https://www.youtube.com/watch?v=31UoHKpyWZw",
        # "https://www.youtube.com/watch?v=pW2AcF4xZAk",
        # "https://www.youtube.com/watch?v=rFuSFCEypaE",
        # "https://www.youtube.com/watch?v=jFby3IEs9V0",
        # "https://www.youtube.com/watch?v=QEuMtD0Skdw",
        # "https://www.youtube.com/watch?v=zLlLtmgZ_KY",
        # "https://www.youtube.com/watch?v=yUocO69hjKY",
        # "https://www.youtube.com/watch?v=gdVr2UrvYnc",
        # "https://www.youtube.com/watch?v=eIRhsl1TDiU",
        # "https://www.youtube.com/watch?v=jQkc_1pQd3g",
        # "https://www.youtube.com/watch?v=2xMFM4eQgbo",
        # "https://www.youtube.com/watch?v=IrHfb32_aiQ",
        # "https://www.youtube.com/watch?v=5FlUYCvyjRI",
        # "https://www.youtube.com/watch?v=VollSfNQAZE",
        # "https://www.youtube.com/watch?v=kV7nA02LfQs",
        # "https://www.youtube.com/watch?v=P9TIErYYwQw",
        # "https://www.youtube.com/watch?v=wys4t-9HE40",
        # "https://www.youtube.com/watch?v=tuYs5UZdmFI",
        # "https://www.youtube.com/watch?v=Ch-MofMT5kI",
        # "https://www.youtube.com/watch?v=CptnIVDhXAY",
        # "https://www.youtube.com/watch?v=RQVGX125QDk",
        # "https://www.youtube.com/watch?v=w97zD6hr9gw",
        # "https://www.youtube.com/watch?v=bd6z20fk5ZE",
        # "https://www.youtube.com/watch?v=2wa1PzBECoE",
        # "https://www.youtube.com/watch?v=oM35apM955A",
        # "https://www.youtube.com/watch?v=IaBFxqJrPiU",
        # "https://www.youtube.com/watch?v=-QvWpWm_PEg",
        # "https://www.youtube.com/watch?v=gfBQ9AqxjuM",
        # "https://www.youtube.com/watch?v=jyayNr65OHc",
        # "https://www.youtube.com/watch?v=Wqiwn6ZNM40",
        # "https://www.youtube.com/watch?v=rpU5-fTxadM",
        # "https://www.youtube.com/watch?v=w_dlGgGFhO4",
        # "https://www.youtube.com/watch?v=fq4OJMcRjsk"
    ],  
}


# olmo = AutoModelForCausalLM.from_pretrained("allenai/OLMo-7B", torch_dtype=torch.float16)
# olmo = olmo.to('cuda')
# tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-7B")

# def run_olmo(messages):
#     messages = [message["content"] for message in messages]

#     inputs = tokenizer(messages, return_tensors='pt', return_token_type_ids=False).to('cuda')
#     response = olmo.generate(**inputs, max_new_tokens=300, do_sample=True, top_p=0.6)
#     return tokenizer.batch_decode(response, skip_special_tokens=True)[0]

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(
    api_key=OPENAI_API_KEY,
)

def run_openai(messages):
    
    response = client.chat.completions.create(
        #model="gpt-3.5-turbo",
        model="gpt-4-0125-preview",
        messages=messages,
        temperature=0.7,
    )

    return [{
        "role": response.choices[0].message.role,
        "content": response.choices[0].message.content,
    }]

def run_llm(messages):
    return run_openai(messages)

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
    frame_paths = []
    if not os.path.exists(video_path):
        print(f"Video file '{video_path}' does not exist.")
        return frame_paths

    if not os.path.exists(f"{video_path}_frames"):
        os.makedirs(f"{video_path}_frames")
    else:
        print(f"Frames for video '{video_path}' already exist.")
        for file in os.listdir(f"{video_path}_frames"):
            frame_paths.append(f"{video_path}_frames/{file}")
        frame_paths = sorted(frame_paths, key=lambda x: int(x.split("/")[-1].split(".")[0]))
        return frame_paths
    video_cap = cv2.VideoCapture(video_path)

    ### save frame at each second
    seconds = 0
    while True:
        video_cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000 + 500)
        res, frame = video_cap.read()
        if (res == False):
            break
        frame_path = f"{video_path}_frames/{seconds}.jpg"
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
        seconds += 1

    video_cap.release()
    return frame_paths


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
    video_title = re.split(r"[/=]", video_link)[-1]
    video_path = os.path.join(DATABASE, f'{video_title}.mp4')
    subtitles_path = os.path.join(DATABASE, f'{video_title}.en.vtt')
    audio_path = os.path.join(DATABASE, f'{video_title}.mp3')

    if os.path.exists(video_path) is False:
        download_video(video_link)
    
    # video_title = metadata.get('id')
    # print(f"'{video_title}'")

    # video_library[video_title] = metadata

    video_frame_paths = extract_frames(video_path)
    subtitles = extract_transcript(subtitles_path, audio_path)

    return video_title, video_frame_paths, subtitles

def tokenize_video(video_link):
    video_title, video_frame_paths, subtitles = process_video(video_link)

    return {
        "subtitles": subtitles,
    }

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

def combine_messages(system_message, user_messages, assistant_messages):
    messages = [{
        "role": "system",
        "content": system_message,
    }]
    last_message = min(len(user_messages), len(assistant_messages))
    for i in range(last_message):
        messages.append({
            "role": "user",
            "content": user_messages[i],
        })
        messages.append({
            "role": "assistant",
            "content": assistant_messages[i],
        })
    
    if last_message < len(user_messages):
        messages.append({
            "role": "user",
            "content": user_messages[last_message],
        })
    return messages

def common_steps_messages_generate(narrations):
    system_message = """
You are an assistant that identifies the common steps in the narrated content across different cooking videos, generates comprehensive steps that occur in each narration, and segments each narration into identified steps.
----------------------------------------
    """
    user_messages = []
    user_messages.append("")
    
    for narration in narrations:
        id = narration["id"]
        content = narration["content"]
        user_messages[-1] += f"""
Narration {id}: {content}
----------------------------------------
        """

    user_messages[-1] += """
Given the above narration transcripts, identify the common steps in the narrated content across different cooking videos. Consider the recurring actions, instructions, or procedures that are essential to the cooking process and are consistently present in each narration. If needed, include unique steps that are specific to certain videos to comprehensivly cover each video.

Let's do it step-by-step.
----------------------------------------
    """

    ### Step 1: Describe the steps in each narration comprehensively focusing on commonalities and unqiueness
    user_messages[-1] += """
1. Describe the steps in each narration comprehensively, focusing on commonalities and uniqueness. Identify the recurring actions, instructions, or procedures that are essential to the cooking process and are consistently present in each narration. Include unique steps that are specific to certain videos to comprehensively cover each video.
----------------------------------------
    """

    ### Step 2: Define the list of steps that covera all the narrations
    user_messages.append("""
2. Define the list of steps that cover all the narrations. Make sure that the steps are comprehensive, distinct, and reflect the commonalities and uniqueness of the narrated content.                         
Follow the JSON format below and surround your JSON output with <result></result> tags:
<result>
{
    <step>: {
        "label": <label of the step>,
        "description": <description of the step>,
        "examples": <examples of the step>
    }
}
</result>
----------------------------------------
    """)

    ### segment each narration into identified steps
    return system_message, user_messages


def generate_common_steps(task_id, narrations):
    ### return if file already exists
    if os.path.exists(f"{OUTPUT_FOLDER}/{task_id}/common_steps.json"):
        with open(f"{OUTPUT_FOLDER}/{task_id}/common_steps.json", "r") as f:
            return json.load(f)
    
    system_message, user_messages = common_steps_messages_generate(narrations)
    
    assistant_messages = []
    messages = combine_messages(system_message, user_messages, assistant_messages)

    ### Step 1: Describe the steps in each narration comprehensively focusing on commonalities and unqiueness
    response = run_llm(messages)
    assistant_messages.append(response[-1]["content"])
    
    ### Step 2: Define the list of steps that covera all the narrations
    messages = combine_messages(system_message, user_messages, assistant_messages)
    response = run_llm(messages)
    assistant_messages.append(response[-1]["content"])

    ### extract list of steps
    # common_steps = json.loads(response[-1]["content"])

    ### Step 3: Segment each narration into identified steps
    
    assign_user_message = """
3. Segment each narration into identified steps based on the defined list of steps. For each step, output the corresponding narration segment that align with the step. If there is no narration segment that aligns with a step, leave it empty.

Ensure that the segmentation is accurate, comprehensive, and reflects the commonalities and uniqueness of the narrated content. Briefly explain the segementation for each step.

Follow the JSON format below and surround your JSON output with <result></result> tags:
<result>
{
    <step>: {
        "label": <label of the step>,
        "narration": <segment of narration transcript>,
        "explanation": <explanation of the segmentation>
    }
}
</result>
----------------------------------------
"""

    segmentations = []
    for narration in narrations:
        id = narration["id"]
        content = narration["content"]
        user_message = f"""
Narration: {content}
----------------------------------------
{assign_user_message}
        """
        messages = combine_messages(system_message, user_messages + [user_message], assistant_messages)
        response = run_llm(messages)

        segmentations.append({
            "id": id,
            "content": content,
            "segmentation": response[-1]["content"]
        })
    combine_messages(system_message, user_messages, assistant_messages)
    ### Print all prev. messages
    for message in messages:
        print(f"{message['role']}: {message['content']}")

    ### Print the final segmentations
    for segmentation in segmentations:
        print(f"Narration {segmentation['id']}: {segmentation['content']}")
        print(f"Segementation: {segmentation['segmentation']}")
        print("--------------------------------------------------")

    narrations_per_step = {}
    for segmentation in segmentations:
        json_string = segmentation["segmentation"].split('<result>')[1].split('</result>')[0]
        json_object = json.loads(json_string.strip())
        for step, segment in json_object.items():
            if step not in narrations_per_step:
                narrations_per_step[step] = []
            narrations_per_step[step].append({
                "id": segmentation["id"],
                "content": segment["narration"],
                "label": segment["label"],
                "explanation": segment["explanation"]
            })


    ### dump the narrations per step
    with open(f"{OUTPUT_FOLDER}/{task_id}/common_steps.json", "w") as f:
        json.dump(narrations_per_step, f, indent=2)

    return narrations_per_step

def variability_messages_generate(narrations):
    system_message = """
You are an assistant that identifies the variability in the narrated content across different cooking videos, defines variability dimensions, and assigns dimension values for each narration transcript.
----------------------------------------
    """
    user_messages = []
    user_messages.append("")
    
    for narration in narrations:
        id = narration["id"]
        content = narration["content"]
        user_messages[-1] += f"""
Narration {id}: {content}
----------------------------------------
        """

    user_messages[-1] += """
Given the above narration transcripts, identify the variability in the narrated content across different cooking videos. Consider the differences in the level of detail, explanation, and personalization in the narration, and how these variations impact the viewer's understanding and engagement with the content.

Let's do it step-by-step.
----------------------------------------
    """

    ### Step 1: Generate summary of differences
    user_messages[-1] += """
1. Povide a summary of the observed differences, highlighting the key distinctions, their implications for the overall viewing experience, and 
their utility for navigation accross videos.
----------------------------------------
    """

    ### Step 2: Define variability dimensions
    user_messages.append("""
2. Define variability dimensions that capture the differences identified earlier. Make sure that the dimensions are comprehensive, distinct, and relect the implications on overall viewing experience, and utility for navigation across videos.
Follow the JSON format below and surround your JSON output with <result></result> tags:
<result>
{
    <dimension>: {
        "label": <label of the dimension>,
        "description": <description of the dimension>,
        "values": <values of the dimension>,
        "examples": <examples of the dimension>
    }
}
</result>
----------------------------------------
    """)

    ### assign dimension values for each narration
    # message generated for each narration

    return system_message, user_messages

def generate_variability(task_id, narrations):
    system_message, user_messages = variability_messages_generate(narrations)
    
    assistant_messages = []
    messages = combine_messages(system_message, user_messages, assistant_messages)

    ### Step 1: Generate summary of differences
    response = run_llm(messages)
    assistant_messages.append(response[-1]["content"])
    
    ### Step 2: Define variability dimensions
    messages = combine_messages(system_message, user_messages, assistant_messages)
    response = run_llm(messages)
    assistant_messages.append(response[-1]["content"])

    ### extract list of variability dimensions
    json_string = response[-1]["content"].split('<result>')[1].split('</result>')[0]
    variability_dimensions = json.loads(json_string.strip())

    ### save the variability dimensions
    with open(f"{OUTPUT_FOLDER}/{task_id}/variability_dimensions.json", "w") as f:
        json.dump(variability_dimensions, f, indent=2)

    ### Step 3: Assign dimension values
    
    assign_user_message = """
3. Assign dimension values for the given narration transcript based on the defined variability dimensions. For each variability dimensions, identify the most appropriate value of variability and provide a brief explanation why the value is appropriate.
Follow the JSON format below and surround your JSON output with <result></result> tags:
<result>
{
    <dimension>: {
        "label": <label of the dimension>,
        "value": <value of the dimension>,
        "explanation": <explanation/reasoning of the value>
    }
}
</result>
----------------------------------------
"""

    assignments = []
    for narration in narrations:
        id = narration["id"]
        label = narration["label"]
        explanation = narration["explanation"]
        content = narration["content"]
        user_message = f"""
Narration: {content}
----------------------------------------
{assign_user_message}
        """
        messages = combine_messages(system_message, user_messages + [user_message], assistant_messages)
        response = run_llm(messages)

        assignments.append({
            "id": id,
            "content": content,
            "label": label,
            "explanation": explanation,
            "assignment": response[-1]["content"]
        })
    combine_messages(system_message, user_messages, assistant_messages)
    ### Print all prev. messages
    for message in messages:
        print(f"{message['role']}: {message['content']}")

    ### Print the final assignments
    for assignment in assignments:
        print(f"Narration {assignment['id']}: {assignment['content']}")
        print(f"Assignment: {assignment['assignment']}")
        print("--------------------------------------------------")

    parsed_assignments = []
    for assignment in assignments:
        json_string = assignment["assignment"].split('<result>')[1].split('</result>')[0]
        json_object = json.loads(json_string.strip())
        parsed_assignments.append({
            "id": assignment["id"],
            "content": assignment["content"],
            "label": assignment["label"],
            "step_assignment_explanation": assignment["explanation"],
            "assignment": json_object
        })
    
    return parsed_assignments
    

def simple_pipeline(task_id):
    ### check if OUTPUT_FOLDER exists
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
    ### check if OUTPUT_FOLDER/task_id exists
    if not os.path.exists(f"{OUTPUT_FOLDER}/{task_id}"):
        os.makedirs(f"{OUTPUT_FOLDER}/{task_id}")

    narrations = []

    for video_link in EMBEDDING_VIDEOS[task_id]:
        result = tokenize_video(video_link)
        combined_subtitles = ""
        for subtitle in result["subtitles"]:
            combined_subtitles += f"{subtitle['text']}\n"

        # remove new lines
        combined_subtitles = combined_subtitles.replace("\n", " ")
        # remove any extra spaces
        combined_subtitles = re.sub(' +', ' ', combined_subtitles)

        narrations.append({
            "id": len(narrations),
            "content": combined_subtitles,
        })

    narrations_per_step = generate_common_steps(task_id, narrations)

    all_assignments = {}

    # for step in narrations_per_step:
    #     print(f"Step: {step}")
    #     new_narrations = []
    #     for narration in narrations_per_step[step]:
    #         new_narrations.append({
    #             "id": narration["id"],
    #             "content": narration["content"],
    #         })
    #     assigment = generate_variability(new_narrations)
    #     # update  
    #     all_assignments.append({
    #         "step": step,
    #         "assignments": assigment
    #     })
    #     ### dump the assignments
    #     with open(f"{OUTPUT_FOLDER}/{step}_variability_assignments.json", "w") as f:
    #         json.dump(assigment, f, indent=2)

    all_narrations = []
    for step in narrations_per_step:
        for narration in narrations_per_step[step]:
            unique_id = f"{step}_{narration['id']}"
            if narration["content"] == "":
                continue
            all_narrations.append({
                "id": unique_id,
                "label": narration["label"],
                "explanation": narration["explanation"],
                "content": narration["content"],
            })

    variability_assignments = generate_variability(task_id, all_narrations)

    for assignment in variability_assignments:
        unique_id = assignment["id"]
        step_id, narration_id = unique_id.split("_")
        narration_id = int(narration_id)
        if narration_id not in all_assignments:
            all_assignments[narration_id] = {}
        all_assignments[narration_id][step_id] = {
            "content": assignment["content"],
            "label": assignment["label"],
            "step_assignment_explanation": assignment["step_assignment_explanation"],
            "assignment": assignment["assignment"]
        }

    ### dump all assignments
    with open(f"{OUTPUT_FOLDER}/{task_id}/assignments.json", "w") as f:
        json.dump(all_assignments, f, indent=2)

def main():
    # annotate_videos()
    # classify_videos()
    task_ids = [
        "11967",
        "13630",
    ]
    simple_pipeline(task_ids[1])

if __name__ == "__main__":
    main()
