from pydantic import BaseModel, Field

class SummarizedAlignmentSchema(BaseModel):
    title: str = Field(..., title="the 5-word title of the new content in the current video.")
    description: str = Field(..., title="a brief, specific, and clear description of the new procedural content. Focus on one specific point at a time, avoid combining multiple details.")
    reasoning: str = Field(..., title="briefly explain why this specific content is in the video. Why is this included in the video?")
    comparison: str = Field(..., title="if applicable, explain why this content is different from or not included in the other video.")

class NotableInformationSchema(BaseModel):
    title: str = Field(..., title="the concise title of the notable information that is present in the current video.")
    description: str = Field(..., title="the description of the notable information.")

class HookSchema(BaseModel):
    title: str = Field(..., title="the title of the hook in a conversational manner. It should be interesting and engaging, but short!")
    description: str = Field(..., title="the elaboration on the hook. It should look like continuation of the title.")