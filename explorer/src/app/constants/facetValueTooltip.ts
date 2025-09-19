export const FACET_VALUE_PERCENTAGE_EXPLANATION =
  "Information coverage is {infoCoverage}%.\n\n" +
  "This shows the coverage for the currently selected information type in the Statistics panel.\n" +
  "• If 'All Types' is selected: average coverage across all information types\n" +
  "• If a specific type is selected: coverage for that specific type only\n\n" +
  "The percentage represents the proportion of contexts (facet value combinations) that match the current facet selections along with this condition and contain at least one piece of information of the relevant type(s).\n\n" +
  "The denominator can be set to 'Theoretical' (all possible combinations) or 'Relative' (only combinations that exist in the data)."

export const DENOMINATOR_TOOLTIP =
  "Theoretical: all the distinct contexts (i.e., facet condition combinations) **theoretically possible** that match current selections.\n\n" + 
  "Relative: all the distinct contexts (i.e., facet condition combinations) **in the data** that match current selections.";

export const NO_CONTEXTS_MESSAGE = "No such context in the data that matches the current facet selections along with the current condition.";
