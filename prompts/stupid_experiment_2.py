from helpers import get_response_pydantic

from pydantic_models.experiment_2 import StepListSchema, TaxonomySchema

from pydantic_models.experiment_2 import IPOListSchema, IPOTaxonomySchema

from pydantic_models.experiment_2 import IPOInformationSchema

from pydantic_models.experiment_2 import ClusterInformationSchema

SYSTEM_PROMPT_TASK = """
You are an expert in analyzing procedural instructional content for the task `{task}`.
"""

USER_PROMPT_EXTRACT_STEPS = """
Given the instructional text (e.g., article, video transcript, etc.) extract the key high-level steps involved in the task.
<instructions>
Follow these guidelines when extracting steps:
1. Steps should be high-level and concise.
2. Base each step on an intermediate outcome with tangible results (e.g., "Make Dough", "Grill Steak"), instead of individual actions (e.g., "Add Flour", "Turn on Grill").
3. Avoid using specific ingredients in the step name (e.g., "Add Tomato Paste"). Instead, focus on the purpose of the step (e.g., "Make Sauce" instead of "Add Tomato Paste").
4. Group together related low-level actions into a single, high-level step. (e.g., combine "Add Salt" and "Add Lime" into "Make Sauce").
5. A step must span multiple sentences, not just a single sentence. It should be sufficiently high-level.
6. Use a concise "verb + object" format to describe each step, containing only one verb (e.g., "Boil Potatoes").
7. Exclude any steps unrelated to the core task, such as introductions, conclusions, or general commentary.
</instructions>
Return a series of concise, high-level steps as a list.
"""

def extract_steps(task, tutorial):

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_TASK.format(task=task),
        },
        {
            "role": "user",
            "content": USER_PROMPT_EXTRACT_STEPS,
        },
        {
            "role": "user",
            "content": f"<tutorial>\n{tutorial}\n</tutorial>"
        },
    ]

    response = get_response_pydantic(messages, StepListSchema)
    steps = response["steps"]
    return steps


USER_PROMPT_AGGREGATE_STEPS = """
Given the list of steps, aggregate them into a taxonomy.
<instructions>
1. Analyze each of the steps and determine the phase of the task that each step belongs to. Each phase should be high-level and include multiple different steps.
2. Group the steps according to the phase.
3. For each phase, generate representative steps that comprehensively cover the range of steps in the phase.
4. Return the taxonomy according to the specifiedformat.
</instructions>
"""


def aggregate_steps_stupid(task, steps):

    steps_str = "\n".join([f"{step['step']}: {step['description']}" for step in steps])

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_TASK.format(task=task),
        },
        {
            "role": "user",
            "content": USER_PROMPT_EXTRACT_STEPS,
        },
        {
            "role": "user",
            "content": f"<steps>\n{steps_str}\n</steps>"
        },
    ]

    response = get_response_pydantic(messages, TaxonomySchema)
    phases = response["phases"]
    step_taxonomy = []
    for phase in phases:
        for step in phase["steps"]:
            step_taxonomy.append({
                "step": step["step"],
                "description": step["description"],
                "phase": phase["phase"],
            })
    return step_taxonomy


USER_PROMPT_EXTRACT_IPOS = """
You are given an instructional text (e.g., article, video transcript, tutorial, etc.) along with a list of steps. Your task is to extract IPOs — Inputs, Outputs, and Instructions — for each step.

<instructions>
1. Check step presence: Determine whether each step is explicitly or implicitly present in the instructional text.
2. If the step is present, extract the following:
    2-1. Instructions: A list of atomic operations describing what is being done in the step.
        - Break compound instructions into separate actions.
        - Keep each instruction clear, concise, and standalone.
    2-2. Inputs: All materials, ingredients, tools, conditions, or data required for the step to proceed.
        - Express inputs in basic noun, noun phrase, or adjective form only.
        - Do not include modifiers (e.g., “2 tablespoons of hot olive oil” → just "oil").
        - If inputs are implied (e.g., a pan for frying), infer them.
    2-3. Outputs: All resulting products, changes, or outcomes of the step.
        - Follow same formatting rules as inputs.
        - Use consistent naming across steps (e.g., if one step outputs "dough", another step should use "dough" as an input, not "mixture").
3. If the step is not present, specify that it is absent.
</instructions>
Return the IPOs for each step in the specified format.
"""

def extract_ipos(task, steps, tutorial):

    steps_str = "\n".join([f"{step['step']}: {step['description']}" for step in steps])

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_TASK.format(task=task),
        },
        {
            "role": "user",
            "content": USER_PROMPT_EXTRACT_IPOS,
                },
        {
            "role": "user",
            "content": f"<steps>\n{steps_str}\n</steps>"
        },
        {
            "role": "user",
            "content": f"<tutorial>\n{tutorial}\n</tutorial>"
        },
    ]

    response = get_response_pydantic(messages, IPOListSchema)
    ipos = response["ipos"]

    ### check if the step is present in the steps list
    filtered_ipos = []
    for ipo in ipos:
        for step in steps:
            if ipo["step"] == step["step"]:
                filtered_ipos.append(ipo)
                break
    return filtered_ipos


USER_PROMPT_TAXONOMIZE_IPOS = """
You are given a summaries of inputs-instructions-outputs for a subtask {subtask} from {n_tutorials} tutorials.
<instructions>
1. Analyze the each part of the summaries (i.e., inputs, instructions, outputs) and consolidate them into a respective and comprehensive taxonomies.
    1-1. For inputs, identify and categorize the high-level roles/types/purposes of the inputs with respect to the subtask.
    1-2. For instructions, identify and categorize the high-level methods/techniques/strategies used to perform the subtask. Each set of instructions must belong to a single category.
    1-3. For outputs, identify and categorize the high-level roles/types/purposes of the outputs in the overall task.
2. For each category in the taxonomies, return five representative examples referring to the summaries.
</instructions>
Return the taxonomies in the specified format.
"""
def taxonomize_ipos_stupid(task, ipos_per_tutorial, subtask):

    ipos_str = ""
    tutorial_count = len(ipos_per_tutorial)
    for tutorial_idx, tutorial in enumerate(ipos_per_tutorial):
        ipos_str += f"<tutorial-{tutorial_idx+1}>\n"
        for ipo_key in tutorial:
            ipos_str += f"<{ipo_key}>\n"
            for ipo in tutorial[ipo_key]:
                ipos_str += f"{ipo}; "
            ipos_str += f"</{ipo_key}>\n"
        ipos_str += f"</tutorial-{tutorial_idx+1}>\n"

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_TASK.format(task=task),
        },
        {
            "role": "user",
            "content": USER_PROMPT_TAXONOMIZE_IPOS.format(subtask=subtask, n_tutorials=tutorial_count),
        },
        {
            "role": "user",
            "content": f"<ipos>\n{ipos_str}\n</ipos>"
        },
    ]

    response = get_response_pydantic(messages, IPOTaxonomySchema)
    taxonomies = {
        "inputs": response["inputs"],
        "methods": response["methods"],
        "outputs": response["outputs"], 
    }
    return taxonomies



"""
    2-1. For inputs or outputs:
        a) their detailed but concise description
        b) any useful information about the subject (e.g., alternatives, substitutes, etc.)
    2-2. For methods:
        a) their detailed but concise description
        b) any useful information about the method (e.g., alternatives, substitutes, etc.)
        c) the explanation about the method (e.g., reasons why it was performed and its consequences)
        d) the tips/warnings about the method


##d) any useful information about the subject (e.g., alternatives, substitutes, etc.) according to the instructional text

Follow the format similar to the examples of the `subject of interest` in the taxonomy.
"""

### TODO: too much explanations are happening...

USER_PROMPT_EXTRACT_INFORMATION_PER_SUBJECT = """
Given the tutorial (e.g., article, video transcript, etc.) and the taxonomy of `subjects of interest` per subtask {subtask}, extract the useful information for each `subject of interest` from the tutorial or specify that it is absent. There are three types of `subjects of interest`: inputs, outputs, and methods.
<instructions>
Let's go step by step:
1. Identify if the `subject of interest` is present or mentioned in the tutorial.
2a. If the `subject of interest` is present, scan the tutorial and extract following information IF AND ONLY IF it is mentioned in the tutorial.
    2a-1. For inputs and outputs extract:
        a) the detailed description of the inputs/outputs according to the tutorial. 
    2a-2. For methods extract:
        a) the concise, but detailed instruction about the method according to the tutorial.
        b) the explanation about the `subject of interest` (e.g., reasons why it was performed and its consequences) according to the tutorial. 
        c) the tips (e.g., enhance the efficiency, improve the quality, etc.) or warnings (e.g., actions to avoid, etc.) about the `subject of interest` according to the tutorial.
2b. If the `subject of interest` is not present, specify that it is absent and do not extract any information.
3. Return the information for each `subject of interest` in the specified format.
</instructions>
{example}
"""

def extract_information_per_ipo_stupid(task, ipo_taxonomy, tutorial, subtask):
    subjects_of_interest_str = ""

    for subject_type in ipo_taxonomy:
        subjects_of_interest_str += f"<{subject_type}>\n"
        for subject in ipo_taxonomy[subject_type]:
            subjects_of_interest_str += f"\t<subject>\n"
            subjects_of_interest_str += f"\t\t<name> {subject['name']} </name>\n"
            subjects_of_interest_str += f"\t\t<definition> {subject['description']} </definition>\n"
            if len(subject["examples"]) > 0:
                subjects_of_interest_str += f"\t\t<examples> {'; '.join(subject['examples'])} </examples>\n"
            subjects_of_interest_str += f"\t<\subject>\n"
        subjects_of_interest_str += f"</{subject_type}>\n"

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_TASK.format(task=task),
        },
        {
            "role": "user",
            "content": USER_PROMPT_EXTRACT_INFORMATION_PER_SUBJECT.format(subtask=subtask, example=EXAMPLE_EXTRACT_INFORMATION_PER_SUBJECT),
        },
        {
            "role": "user",
            "content": f"<subjects_of_interest>\n{subjects_of_interest_str}\n</subjects_of_interest>"
        },
        {
            "role": "user",
            "content": f"<tutorial>\n{tutorial}\n</tutorial>"
        },
    ]

    response = get_response_pydantic(messages, IPOInformationSchema)

    def remove_refs(cur_list):
        new_list = []
        for item in cur_list:
            if item["ref"] != "":
                new_list.append(item["text"])
        return new_list

    for input in response["inputs"]:
        input["description"] = [input["description"]]
        
        input["instruction"] = []
        input["explanations"] = []
        input["tips"] = []

    for output in response["outputs"]:
        output["description"] = [output["description"]]

        output["instruction"] = []
        output["explanations"] = []
        output["tips"] = []

    for method in response["methods"]:
        method["instruction"] = [method["instruction"]]
        method["description"] = []
        method["explanations"] = remove_refs(method["explanations"])
        method["tips"] = remove_refs(method["tips"])

    information = {
        "inputs": response["inputs"],
        "methods": response["methods"],
        "outputs": response["outputs"],
    }
    return information


EXAMPLE_EXTRACT_INFORMATION_PER_SUBJECT = """
<example>
    <subjects_of_interest>
        <inputs>
            <subject>
                <name> Base Ingredients </name>
                <definition> Core dry and wet ingredients fundamental to muffin making </definition>
                <examples> flour; sugar; eggs; milk; baking powder </examples>
            <\subject>
            <subject>
                <name> Flavor Enhancers </name>
                <definition> Ingredients that add unique flavors and textures to muffins </definition>
                <examples> chocolate chips; blueberries; bananas; fruits; peanut butter </examples>
            <\subject>
            <subject>
                <name> Spices and Seasonings </name>
                <definition> Aromatic ingredients that provide depth and complexity </definition>
                <examples> cinnamon; ginger; nutmeg; cloves; salt </examples>
            <\subject>
            <subject>
                <name> Binding and Moisture Agents </name>
                <definition> Ingredients that help bind and maintain moisture in muffins </definition>
                <examples> vegetable oil; canola oil; milk; eggs; cream cheese </examples>
            <\subject>
            <subject>
                <name> Nutritional Boosters </name>
                <definition> Ingredients that add nutritional value and texture </definition>
                <examples> flaxseed; chia seeds; whole oats; maple syrup; egg yolk </examples>
            <\subject>
        </inputs>
        <methods>
            <subject>
                <name> Ingredient Collection </name>
                <definition> Process of gathering all required ingredients for muffin preparation </definition>
                <examples> Gather ingredients from pantry; Check ingredient availability; Collect all specified ingredients; Organize ingredients before preparation; Ensure all ingredients are fresh </examples>
            <\subject>
            <subject>
                <name> Precise Measurement </name>
                <definition> Accurate measuring of ingredients to ensure consistent results </definition>
                <examples> Measure dry ingredients; Weigh ingredients precisely; Use measuring cups and spoons; Level off dry ingredients; Measure liquid ingredients at eye level </examples>
            <\subject>
        </methods>
        <outputs>
            <subject>
                <name> Prepared Ingredient Set </name>
                <definition> Fully measured and organized ingredients ready for further processing </definition>
                <examples> measured ingredients; weighed and sorted ingredients; precisely portioned ingredients; organized ingredient mise en place; ingredient set prepared for mixing </examples>
            <\subject>
        </outputs>
    </subjects_of_interest>

    <tutorial>
        Easy Pumpkin Cheesecake Muffins

        The perfect combination of spicy pumpkin and dreamy cheesecake married together in a bite-sized muffin.  It's the treat you've been waiting for!

        These soft muffins with a saucy swirl of cheesecake make for the perfect treat to bring to all of your holiday gatherings.  Whether your into jumbo muffins, or bite-sized minis, these muffins are sure to be an instant hit !

        Step 1: Ingredients
        TOOLS:
        Mini Muffin Tin
        Mixer
        Digital Scale - I like this one

        MUFFINS:
        1 15 oz can pumpkin
        1/4 cup vegetable oil
        2 large eggs
        1 (200g) cup sugar
        1 teaspoon vanilla
        1 1/2 cups (188g) all-purpose flour
        1 teaspoon baking powder
        1/2 teaspoon baking soda
        1/2 teaspoon ground cinnamon
        1/2 teaspoon ground ginger
        1/2 teaspoon ground nutmeg
        1/8 teaspoon ground cloves
        1/2 teaspoon salt
        1/4 teaspoon freshly ground black pepper
        FILLING:
        8 ounces (200g) cream cheese, at room temperature
        1 large egg yolk
        5 tablespoons (75g) sugar
        1/8 teaspoon vanilla extract
        Chopped nuts (walnuts, pecans) (optional)

        Step 2: Muffins
        Preheat oven to 350F (180C)

        Mix canned pumpkin, oil, and sugar together, either by hand or with a mixer on low.
        Add in eggs, one at a time, combining thoroughly after each.
        Add vanilla.

        Whisk together flour, baking powder, baking soda, salt, and spices.

        Slowly add flour mixture to liquids, combining thoroughly and scraping the sides of the bowl with a spatula until well combined.

        Step 3: Filling
        Mix together cream cheese, sugar, egg yolk and vanilla until well combined.

        Step 4: Assemble
        Grease your muffin tins or line with paper cups. 

        Drop in spoonfuls of pumpkin mixture.
        Add smaller spoonfuls of cream cheese mixture on top.
        Swirl with a skewer.

        Optional:  top with crushed nuts.  Pumpkin seeds (pepitas) are also a great topping!

        Step 5: Bake
        Bake as follows, turning the tins half way through baking times:

        mini muffins: 20-25 min
        standard muffins: 25-30 min
        jumbo muffins: 30-40 min


        Wasn't that easy?

        Enjoy!
    </tutorial>

    <response>
    {
        "inputs": [
            {
                "name": "Base Ingredients",
                "present": true,
                "description": "1 15 oz can pumpkin, 1/4 cup vegetable oil, 2 large eggs, 1 cup sugar, 1 teaspoon vanilla, 1 1/2 cups all-purpose flour, 1 teaspoon baking powder, 1/2 teaspoon baking soda"
            },
            {
                "name": "Flavor Enhancers",
                "present": false,
                "description": ""
            },
            {
                "name": "Spices and Seasonings",
                "present": true,
                "description": "1/2 teaspoon ground cinnamon, 1/2 teaspoon ground ginger, 1/2 teaspoon ground nutmeg, 1/8 teaspoon ground cloves, 1/2 teaspoon salt, 1/4 teaspoon freshly ground black pepper"
            },
            {
                "name": "Binding and Moisture Agents",
                "present": true,
                "description": "1/4 cup vegetable oil, 2 large eggs, 8 ounces cream cheese, 1 egg yolk"
            },
            {
                "name": "Nutritional Boosters",
                "present": true,
                "description": "optional chopped nuts (walnuts, pecans), optional pumpkin seeds (pepitas)"
            }
        ],
        "methods": [
            {
                "name": "Ingredient Collection",
                "present": true,
                "instruction": "Gather all ingredients listed in the recipe.",
                "explanations": [],
                "tips": [
                    {
                        "ref": "at room temperature",
                        "text": "Ensure cream cheese is at room temperature for easier mixing."
                    }
                ]
            },
            {
                "name": "Precise Measurement",
                "present": true,
                "instruction": "Measure ingredients precisely using digital scale.",
                "explanations": [],
                "tips": []
            }
        ],
        "outputs": [
            {
                "name": "Prepared Ingredient Set",
                "present": true,
                "description": "1 15 oz can pumpkin, 1/4 cup vegetable oil, 2 large eggs, 1 cup sugar, 1 teaspoon vanilla, 1 1/2 cups all-purpose flour, 1 teaspoon baking powder, 1/2 teaspoon baking soda, 1/2 teaspoon ground cinnamon, 1/2 teaspoon ground ginger, 1/2 teaspoon ground nutmeg, 1/8 teaspoon ground cloves, 1/2 teaspoon salt, 1/4 teaspoon freshly ground black pepper, 1/4 cup vegetable oil, 2 large eggs, 8 ounces cream cheese, 1 egg yolk, optional chopped nuts (walnuts, pecans), optional pumpkin seeds (pepitas)"
            }
        ]
    }
    </response>
</example>
"""



USER_PROMPT_CLUSTER_INFORMATION = """
Given the list of information pices of type `{information_type}` about `{subtask}`, cluster them into groups based on their similarity.
<instructions>
Let's do it step by step:
1. Find all the common information pieces in the list.
2. Group the information pieces into clusters.
3. Return the clusters in the specified format: representative information and the ids of associated information pieces. Make sure that each information piece is assigned to exactly one cluster.
</instructions>
{example}
"""

def cluster_information_stupid(task, information_list, subtask, information_type):
    if len(information_list) == 0:
        return []
    if len(information_list) == 1:
        return [0]
    information_str = "<information_list>\n"
    for idx, information in enumerate(information_list):
        information_str += f"\t<information>\n"
        information_str += f"\t\t<id> {idx} </id>\n"
        information_str += f"\t\t<content> {information} </content>\n"
        information_str += f"\t</information>\n"
    information_str += "</information_list>"
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_TASK.format(task=task),
        },
        {
            "role": "user",
            "content": USER_PROMPT_CLUSTER_INFORMATION.format(subtask=subtask, information_type=information_type, example=EXAMPLE_CLUSTER_INFORMATION),
        },
        {
            "role": "user",
            "content": f"{information_str}"
        },
    ]

    response = get_response_pydantic(messages, ClusterInformationSchema)
    result = [-1 for _ in range(len(information_list))]
    for idx, cluster in enumerate(response["clusters"]):
        for id in cluster["ids"]:
            result[id] = idx

    
    ### check if not assigned (i.e., not in any cluster), then assign it to a new cluster
    count = 0
    for idx, information in enumerate(information_list):
        if result[idx] == -1:
            result[idx] = len(response["clusters"])
            count += 1
    if count > 0:
        print(f"Warning: {count} information pieces are not assigned to any cluster.")
    return result


EXAMPLE_CLUSTER_INFORMATION = """
<example>
    The list of information pices of type `description` about `Preparation -> Secure Vehicle Location -> inputs -> Vehicle Type`
    <information_list>
        <information>
            <id> 0 </id>
            <content> car; vehicle; automobile </content>
        </information>
        <information>
            <id> 1 </id>
            <content> 2013 Toyota Corolla </content>
        </information>
        <information>
            <id> 2 </id>
            <content> Honda vehicle </content>
        </information>
        <information>
            <id> 3 </id>
            <content> vehicle; car </content>
        </information>
        <information>
            <id> 4 </id>
            <content> car </content>
        </information>
        <information>
            <id> 5 </id>
            <content> tractor </content>
        </information>
        <information>
            <id> 6 </id>
            <content> Unspecified vehicle (tutorial mentions checking owner's manual for specific vehicle details) </content>
        </information>
        <information>
            <id> 7 </id>
            <content> Chevrolet full-size pickup truck </content>
        </information>
        <information>
            <id> 8 </id>
            <content> car; vehicle; automobile </content>
        </information>
        <information>
            <id> 9 </id>
            <content> vehicle for tire change </content>
        </information>
        <information>
            <id> 10 </id>
            <content> vehicle; car </content>
        </information>
        <information>
            <id> 11 </id>
            <content> car </content>
        </information>
        <information>
            <id> 12 </id>
            <content> GMC Sierra </content>
        </information>
        <information>
            <id> 13 </id>
            <content> Toyota vehicle </content>
        </information>
        <information>
            <id> 14 </id>
            <content> 2015 golf SportWagen </content>
        </information>
        <information>
            <id> 15 </id>
            <content> truck (implied by references to 'bed of the truck') </content>
        </information>
        <information>
            <id> 16 </id>
            <content> vehicle (unspecified type) </content>
        </information>
        <information>
            <id> 17 </id>
            <content> 17 row vehicle (Nissan) </content>
        </information>
        <information>
            <id> 18 </id>
            <content> Jeep vehicle </content>
        </information>
        <information>
            <id> 19 </id>
            <content> 2013 Volkswagen Jetta SE </content>
        </information>
        <information>
            <id> 20 </id>
            <content> vehicle; automobile </content>
        </information>
        <information>
            <id> 21 </id>
            <content> Honda Accord </content>
        </information>
        <information>
            <id> 22 </id>
            <content> Honda CRV </content>
        </information>
        <information>
            <id> 23 </id>
            <content> vehicle; car </content>
        </information>
        <information>
            <id> 24 </id>
            <content> Mercedes vehicle </content>
        </information>
        <information>
            <id> 25 </id>
            <content> Toyota vehicle with rear hatch compartment </content>
        </information>
        <information>
            <id> 26 </id>
            <content> Toyota Corolla </content>
        </information>
        <information>
            <id> 27 </id>
            <content> motorcycle (K 16 bike) </content>
        </information>
        <information>
            <id> 28 </id>
            <content> Honda Jazz test unit </content>
        </information>
        <information>
            <id> 29 </id>
            <content> Mini Cooper </content>
        </information>
        <information>
            <id> 30 </id>
            <content> trailer </content>
        </information>
        <information>
            <id> 31 </id>
            <content> car </content>
        </information>
        <information>
            <id> 32 </id>
            <content> Jeep vehicle (implied by reference to 'Jeep comm slash owners') </content>
        </information>
        <information>
            <id> 33 </id>
            <content> Honda Civic </content>
        </information>
        <information>
            <id> 34 </id>
            <content> car </content>
        </information>
        <information>
            <id> 35 </id>
            <content> car </content>
        </information>
        <information>
            <id> 36 </id>
            <content> Subaru vehicle </content>
        </information>
        <information>
            <id> 37 </id>
            <content> unspecified vehicle (implied by tire change context) </content>
        </information>
        <information>
            <id> 38 </id>
            <content> Jeep vehicle (implied by reference to 'Jeep comm slash owners') </content>
        </information>
        <information>
            <id> 39 </id>
            <content> 2013 Toyota Corolla </content>
        </information>
        <information>
            <id> 40 </id>
            <content> vehicle with scissor jack, tire changing tools under driver's seat, spare tire stowed underneath rear of vehicle </content>
        </information>
        <information>
            <id> 41 </id>
            <content> Honda Civic </content>
        </information>
        <information>
            <id> 42 </id>
            <content> truck, vehicle </content>
        </information>
        <information>
            <id> 43 </id>
            <content> vehicle; automatic vehicle; standard vehicle; truck; older vehicle </content>
        </information>
        <information>
            <id> 44 </id>
            <content> car; Honda </content>
        </information>
        <information>
            <id> 45 </id>
            <content> vehicle (unspecified type) </content>
        </information>
        <information>
            <id> 46 </id>
            <content> car </content>
        </information>
        <information>
            <id> 47 </id>
            <content> car </content>
        </information>
        <information>
            <id> 48 </id>
            <content> car; vehicle; automobile </content>
        </information>
        <information>
            <id> 49 </id>
            <content> Toyota Tacoma truck </content>
        </information>
        <information>
            <id> 50 </id>
            <content> Toyota vehicle </content>
        </information>
        <information>
            <id> 51 </id>
            <content> car </content>
        </information>
        <information>
            <id> 52 </id>
            <content> Chevy Silverado truck </content>
        </information>
        <information>
            <id> 53 </id>
            <content> Honda Civic </content>
        </information>
        <information>
            <id> 54 </id>
            <content> Ford Ranger </content>
        </information>
        <information>
            <id> 55 </id>
            <content> car; vehicle; automobile </content>
        </information>
        <information>
            <id> 56 </id>
            <content> Honda vehicle </content>
        </information>
        <information>
            <id> 57 </id>
            <content> 97 Honda Civic </content>
        </information>
        <information>
            <id> 58 </id>
            <content> vehicle; automobile </content>
        </information>
        <information>
            <id> 59 </id>
            <content> Jeep, passenger car </content>
        </information>
        <information>
            <id> 60 </id>
            <content> full-size truck, lifted truck </content>
        </information>
        <information>
            <id> 61 </id>
            <content> BMW or mini vehicle </content>
        </information>
        <information>
            <id> 62 </id>
            <content> Chevy vehicle </content>
        </information>
        <information>
            <id> 63 </id>
            <content> truck, vehicle with wheel/tire </content>
        </information>
        <information>
            <id> 64 </id>
            <content> car </content>
        </information>
        <information>
            <id> 65 </id>
            <content> Chevy Avalanche </content>
        </information>
        <information>
            <id> 66 </id>
            <content> 2015 Toyota Sienna </content>
        </information>
        <information>
            <id> 67 </id>
            <content> truck </content>
        </information>
        <information>
            <id> 68 </id>
            <content> car </content>
        </information>
        <information>
            <id> 69 </id>
            <content> vehicle with air suspension, truck with jack and tools under front passenger seat </content>
        </information>
        <information>
            <id> 70 </id>
            <content> Jeep </content>
        </information>
        <information>
            <id> 71 </id>
            <content> Ford Ranger pickup truck </content>
        </information>
        <information>
            <id> 72 </id>
            <content> car; vehicle </content>
        </information>
        <information>
            <id> 73 </id>
            <content> vehicle with manual or automatic transmission </content>
        </information>
        <information>
            <id> 74 </id>
            <content> vehicle with a flat tire, vehicle with a trunk, vehicle with a spare tire </content>
        </information>
        <information>
            <id> 75 </id>
            <content> Club Car Precedent golf cart </content>
        </information>
        <information>
            <id> 76 </id>
            <content> car </content>
        </information>
        <information>
            <id> 77 </id>
            <content> unspecified vehicle type </content>
        </information>
        <information>
            <id> 78 </id>
            <content> BMW r1200gs adventure motorcycle </content>
        </information>
        <information>
            <id> 79 </id>
            <content> bicycle </content>
        </information>
        <information>
            <id> 80 </id>
            <content> car; vehicle </content>
        </information>
        <information>
            <id> 81 </id>
            <content> 2000 Volkswagen Beetle </content>
        </information>
        <information>
            <id> 82 </id>
            <content> car with stick-shift </content>
        </information>
        <information>
            <id> 83 </id>
            <content> 2015 Nissan Micra </content>
        </information>
        <information>
            <id> 84 </id>
            <content> 1998 Toyota Camry </content>
        </information>
    </information_list>

    <response>
    {
        "clusters": [
            {
                "representative": "car or passenger vehicle (Toyota, Honda, VW, BMW, Nissan, Mercedes, Subaru, etc.)",
                "ids": [0, 1, 2, 3, 4, 8, 10, 11, 13, 14, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 31, 33, 34, 35, 36, 39, 41, 44, 46, 47, 48, 50, 51, 53, 55, 56, 57, 58, 61, 64, 66, 68, 72, 76, 80, 81, 82, 83, 84]
            },
            {
                "representative": "SUV or off-road passenger vehicle (Jeep, Nissan)",
                "ids": [17, 18, 32, 38, 59, 70]
            },
            {
                "representative": "Pickup trucks (Chevy, GMC, Toyota Tacoma, Ford Ranger)",
                "ids": [7, 12, 15, 42, 43, 49, 52, 54, 60, 62, 63, 65, 67, 71]
            },
            {
                "representative": "Generic or unspecified vehicle, especially in repair or tutorial context",
                "ids": [6, 9, 16, 37, 40, 45, 69, 74, 77]
            },
            {
                "representative": "Motorcycle (BMW, K16, r1200gs)",
                "ids": [27, 78]
            },
            {
                "representative": "Tractor",
                "ids": [5]
            },
            {
                "representative": "Trailer",
                "ids": [30]
            },
            {
                "representative": "Vehicle with manual/automatic transmission (vague)",
                "ids": [73]
            },
            {
                "representative": "Golf cart",
                "ids": [75]
            },
            {
                "representative": "Bicycle",
                "ids": [79]
            }
        ]
    }
    </response>
</example>
"""