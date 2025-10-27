from pydantic import BaseModel, Field

class InformationSchema(BaseModel):
    content: str = Field(..., title="The content of the information.")
    source_doc_idx: int = Field(..., title="The index of the source document in the library.")

class InformationListSchema(BaseModel):
    information_list: list[InformationSchema] = Field(..., title="The list of information pieces retrieved from the library in the order of relevance.")