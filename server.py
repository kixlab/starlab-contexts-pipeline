import json
import cv2

from flask import Flask
from flask_cors import CORS
from flask import request, send_file, redirect, url_for

from pathlib import Path

from helpers.video_scripts import extract_frames

app = Flask(__name__)

CORS(app, origins=["http://localhost:7778", "http://internal.kixlab.org:7778"])
app.config['CORS_HEADERS'] = 'Content-Type'
app.config["UPLOAD_EXTENSIONS"] = [".mp4", ".jpg", ".png", "webm"]

task_ugs = {}

ROOT = Path('.')

DATABASE = ROOT / 'static' / 'database'
RESULTS = ROOT / 'static' / 'results'

def add_urls(links):
    for link in links:
        if "video_id" not in link or "start" not in link:
            continue
        video_id = link["video_id"]
        start = link["start"]
        
        video_path = f"{DATABASE}/{video_id}.mp4"
        link["video_url"] = f"{video_path}"

        frame_paths = extract_frames(video_path)
        ### find the thumbnail with closest time to start
        for sec, frame_path in enumerate(frame_paths):
            if sec >= start:
                link["thumbnail_url"] = f"{frame_path}"
                break
        link["width"] = 1280
        link["height"] = 720
        if len(frame_paths) > 0:
            ### open 0ths frame and get width & height
            frame_path = frame_paths[0]
            frame = cv2.imread(frame_path)
            link["width"] = frame.shape[1]
            link["height"] = frame.shape[0]
            
    return links

@app.route("/get_json_files/<foldername>", methods=["GET"])
def get_json_files(foldername):

    folder = RESULTS / foldername
    files = {}
    for file in folder.iterdir():
        if file.suffix == ".json":
            filename = file.name
            json_path = folder / filename
            with open(json_path, "r") as f:
                data = json.load(f)
                files[filename] = data
    return files

# @app.route("/get_task_info", methods=["POST"])
# def get_task_info():
#     ### {"task_id": "custom"}
#     data = request.json
#     print(data)
#     task_id = data["task_id"]
#     if task_id not in task_ugs:
#         task_ugs[task_id] = setup_ug(task_id)
#     task_ug = task_ugs[task_id]

#     videos_info = []
#     for video in task_ug.videos:
#         segments = list(map(lambda x: {
#             "start": x["start"],
#             "finish": x["finish"],
#             "title": x["title"],
#             "text": x["text"],
#         }, video.common_steps))
#         segments = sorted(segments, key=lambda x: x["start"])
#         videos_info.append({
#             "video_id": video.video_id,
#             "start": 0,
#             "segments": segments,
#         })

#     ### {"videos": {"video_id": string, "start": float, "video_url": string, "thumbnail_url": string}, "request": {"task_id": string}}
#     return {
#         "videos": add_urls(videos_info),
#         "request": data,
#     }

# @app.route("/get_links", methods=["POST"])
# def get_links():
#     data = request.json
#     print(data)
#     # have to get task_id
#     # have to get watch_history [{video_id: string, start: float, finish: float}]
#     task_id = data["task_id"]
#     task_ug = task_ugs[task_id]
#     watch_history = data["watch_history"]
#     link_types = data["link_types"]
#     previous_links = data["links"]
    
#     links = generate_links(task_ug, watch_history, link_types, previous_links)
#     for link_type in links.keys():
#         links[link_type] = add_urls(links[link_type])

#     return {
#         "request": data,
#         "links": links,
#     }

# def test():
#     task_id = "custom"
#     task_ug = setup_ug(task_id)
#     watch_history = []
#     link_types = ["global", "local"]
#     links = generate_links(task_ug, watch_history, link_types)
#     for link_type in links.keys():
#         links[link_type] = add_urls(links[link_type])
#     print(json.dumps(links, indent=4))

def launch_server():
    app.run(host="0.0.0.0", port=7779)

if __name__ == "__main__":
    #test_video("https://www.youtube.com/live/4LdIvyfzoGY?feature=share")
    #test_video("https://youtu.be/XqdDMNExvA0")
    #test_video("https://youtu.be/pZ3HQaGs3uc")
    
    launch_server()
    # test()