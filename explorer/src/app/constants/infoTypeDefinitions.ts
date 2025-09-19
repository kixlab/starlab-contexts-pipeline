export type InfoTypeDefinition = { title: string; text: string };

export const ALL_TYPES_DEFINITION: InfoTypeDefinition = {
  title: "Total Information Coverage",
  text:
    "Average coverage across all types.\n\n" +
    "We compute each type’s % (contexts where it appears / all contexts) and then average those %s.\n\n" +
    "Higher % = more saturation; lower % = bigger gaps.",
};

export const GENERIC_TYPE_DEFINITION: InfoTypeDefinition = {
  title: "Information Type",
  text: "The % of contexts (facet combinations) that include at least one item of this type.\n\n" + 
  "We count presence, not quantity—multiple items in the same context still count once.\n\n" +
  "Higher % = more saturation; lower % = bigger gaps.",
};

export const INFO_TYPE_DEFINITIONS: Record<string, InfoTypeDefinition> = {
  Method: {
    title: "Method Information Coverage",
    text: GENERIC_TYPE_DEFINITION.text,
  },
  Explanation: {
    title: "Explanation Information Coverage",
    text: GENERIC_TYPE_DEFINITION.text,
  },
  Description: {
    title: "Description Information Coverage",
    text: GENERIC_TYPE_DEFINITION.text,
  },
  Supplementary: {
    title: "Supplementary Information Coverage",
    text: GENERIC_TYPE_DEFINITION.text,
  },
};