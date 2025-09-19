export type InfoTypeDefinition = { title: string; text: string };

export const ALL_TYPES_DEFINITION: InfoTypeDefinition = {
  title: "Total Information Coverage",
  text:
    "Average coverage across all information types.\n\n" +
    "Calculation: For each information type, compute (contexts with that type / total contexts), then average these percentages.\n\n" +
    "The denominator can be set to 'Theoretical' (all possible facet combinations) or 'Relative' (only combinations that exist in the data).\n\n" +
    "Higher % = more saturation; lower % = bigger gaps.",
};

export const GENERIC_TYPE_DEFINITION: InfoTypeDefinition = {
  title: "Information Type Coverage",
  text: "The percentage of contexts (facet value combinations) that include at least one item of this specific type.\n\n" + 
  "Calculation: (contexts with this type / total contexts) × 100\n\n" +
  "The denominator can be set to 'Theoretical' (all possible facet combinations) or 'Relative' (only combinations that exist in the data).\n\n" +
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