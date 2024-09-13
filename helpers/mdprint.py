def print_hooks_table(ds, video_id):
    pass

def print_alignments_table(ds, video_id):
    table_prevs = []
    table_unseen = []
    for alignment in ds.alignments:
        if alignment["new_video"] == video_id:
            table_prevs.append({
                "title": alignment["title"],
                "alignments": alignment["alignments"],
                "video_id": alignment["prev_video"]
            })
        elif alignment["prev_video"] == video_id:
            table_unseen.append({
                "title": alignment["title"],
                "alignments": alignment["alignments"],
                "video_id": alignment["new_video"]
            })

    ### print the tables in markdown format where rows are videos and columns are: title, alignments
    for idx, table in enumerate([table_prevs, table_unseen]):
        if idx == 0:
            print("### Table current video vs previous videos")
        else:
            print("### Table current video vs unseen videos")
        print("Video | Title | Alignments |")
        print("| --- | --- | --- |")
        for row in table:
            alignments = "<br>".join([f"<br>**{d_id} -->**<br>" + f"<br>".join([f"-**{key}**: {value}" for key, value in d.items()]) for d_id, d in enumerate(row["alignments"])])
            print(f"| {row['video_id']} | {row['title']} | {alignments} |")
        print("")

def print_video_summaries(video):
    ### print meta_summary and subgoal_summaries
    print(f"### Video {video.video_id}")
    print("#### Meta Summary")
    for key in video.meta_summary:
        if key == "title":
            continue
        print(f"- **{key}**: {video.meta_summary[key]}")
    print("\n\n")
    for subgoal_summary in video.subgoal_summaries:
        print(f"#### Subgoal Summary: {subgoal_summary['title']}")
        for key in subgoal_summary:
            if key == "title":
                continue
            print(f"- **{key}**: {subgoal_summary[key]}")
        print("\n\n")

def print_hooks(hooks_per_class_and_title):
    for classification, hooks_per_title in hooks_per_class_and_title.items():
        print(f"# Classification: {classification}")
        for title, hooks in hooks_per_title.items():
            print(f"## Title: {title}")
            for hook in hooks:
                print(f"### Hook: {hook['title']}")
                print(f"#### NewRef: {hook['newref']}")
                print(f"#### Relevant Alignments")
                for alignment in hook["alignments"]:
                    print(f"Alignment: {alignment['alignment_id']} | {alignment['newref']}: {alignment['description']}\n\n")

                print()
            print()
        print()