export type InfoTypeDefinition = { title: string; text: string };

export const ALL_TYPES_DEFINITION: InfoTypeDefinition = {
  title: "Total Information Coverage",
  text:
    "Average coverage across all information types.\n\n" +
    "Calculation: For each information type, compute (contexts with that type / total contexts), then average these percentages.\n\n" +
    "The denominator can be set to 'Theoretical' (all possible CU-value combinations) or 'Relative' (only combinations that exist in the data).\n\n" +
    "Higher % = more saturation; lower % = bigger gaps.",
};

export const PER_TYPE_COVERAGE_DEFINITION: InfoTypeDefinition = {
  title: "Information Type Coverage",
  text:
    "The percentage of contexts (canonical-value combinations) that include at least one item of this specific type.\n\n" +
    "Calculation: (contexts with this type / total contexts) × 100\n\n" +
    "The denominator can be set to 'Theoretical' (all possible CU-value combinations) or 'Relative' (only combinations that exist in the data).\n\n" +
    "We count presence, not quantity—multiple items in the same context still count once.\n\n" +
    "Higher % = more saturation; lower % = bigger gaps.",
};

// These definitions describe the meaning of information types (not the coverage calculation).
// They are shown in tooltips and in the "Add information type" modal.

const SUBGOAL_DESCRIPTION = `Subgoal: Objective of a subsection.

Example: "Now for the intricate layer that will give me the final webbing look."`;

const INSTRUCTION_DESCRIPTION = `Instruction: Actions that the instructor performs to complete the task.

Example: "We're going to pour that into our silicone baking cups."`;

const TOOL_DESCRIPTION = `Tool: Introduction of the materials, ingredients, and equipment to be used.

Example: "I'm also going to use a pair of scissors, a glue stick, some fancy tape or some regular tape."`;

const METHOD_DESCRIPTION = [SUBGOAL_DESCRIPTION, INSTRUCTION_DESCRIPTION, TOOL_DESCRIPTION].join("\n\n");

const TIP_DESCRIPTION = `Tip: Additional instructions or information that makes instructions easier, faster, or more efficient.

Example: "I find that it's easier to do just a couple of layers at a time instead of all four layers at a time."`;

const WARNING_DESCRIPTION = `Warning: Actions that should be avoided.

Example: "I don't know but I would say avoid using bleach if you can."`;

const SUPPLEMENTARY_DESCRIPTION = [TIP_DESCRIPTION, WARNING_DESCRIPTION].join("\n\n");

const JUSTIFICATION_DESCRIPTION = `Justification: Reasons why the instruction was performed.

Example: "Because every time we wear our contact lenses, makeup and even dirt particles [...] might harm our eyes directly."`;

const EFFECT_DESCRIPTION = `Effect: Consequences of the instruction.

Example: "And these will overhang a little to help hide the gap."`;

const EXPLANATION_DESCRIPTION = [JUSTIFICATION_DESCRIPTION, EFFECT_DESCRIPTION].join("\n\n");

const STATUS_DESCRIPTION = `Status: Descriptions of the current state of the target object.

Example: "Something sticky and dirty all through the back seat."`;

const CONTEXT_DESCRIPTION = `Context: Descriptions of the method or the setting.

Example: "[...] The process of putting on a tip by hand [...] takes a lot of patience but it can be done if you're in a pinch."`;

const TOOL_SPECIFICATION_DESCRIPTION = `Tool Specification: Descriptions of the tools and equipment.

Example: "These are awesome beans, creamy texture, slightly nutty loaded with flavor."`;

const DESCRIPTION_DESCRIPTION = [STATUS_DESCRIPTION, CONTEXT_DESCRIPTION, TOOL_SPECIFICATION_DESCRIPTION].join("\n\n");

const OUTCOME_DESCRIPTION = `Outcome: Descriptions of the final results of the procedure.

Example: "And now we have a dinosaur taggy blanket that wrinkles, so a fun gift for any baby on your gift giving list."`;

const REFLECTION_DESCRIPTION = `Reflection: Summary, evaluation, and suggestions for the future about the overall procedure.

Example: "However, I am still concerned about how safe rubbing alcohol actually is to use so maybe next time, I will give vodka a try."`;

const CONCLUSION_DESCRIPTION = [OUTCOME_DESCRIPTION, REFLECTION_DESCRIPTION].join("\n\n");

export const INFO_TYPE_DEFINITIONS: Record<string, InfoTypeDefinition> = {
  // Coarse
  Method: { title: "Method", text: METHOD_DESCRIPTION },
  Supplementary: { title: "Supplementary", text: SUPPLEMENTARY_DESCRIPTION },
  Explanation: { title: "Explanation", text: EXPLANATION_DESCRIPTION },
  Description: { title: "Description", text: DESCRIPTION_DESCRIPTION },
  Conclusion: { title: "Conclusion", text: CONCLUSION_DESCRIPTION },

  // Fine
  "Method - Subgoal": { title: "Method - Subgoal", text: SUBGOAL_DESCRIPTION },
  "Method - Instruction": { title: "Method - Instruction", text: INSTRUCTION_DESCRIPTION },
  "Method - Tool": { title: "Method - Tool", text: TOOL_DESCRIPTION },

  "Supplementary - Tip": { title: "Supplementary - Tip", text: TIP_DESCRIPTION },
  "Supplementary - Warning": { title: "Supplementary - Warning", text: WARNING_DESCRIPTION },

  "Explanation - Justification": { title: "Explanation - Justification", text: JUSTIFICATION_DESCRIPTION },
  "Explanation - Effect": { title: "Explanation - Effect", text: EFFECT_DESCRIPTION },

  "Description - Status": { title: "Description - Status", text: STATUS_DESCRIPTION },
  "Description - Context": { title: "Description - Context", text: CONTEXT_DESCRIPTION },
  "Description - Tool Specification": { title: "Description - Tool Specification", text: TOOL_SPECIFICATION_DESCRIPTION },

  "Conclusion - Outcome": { title: "Conclusion - Outcome", text: OUTCOME_DESCRIPTION },
  "Conclusion - Reflection": { title: "Conclusion - Reflection", text: REFLECTION_DESCRIPTION },
};
