import json

from PIL import Image
from openai import OpenAI

from helpers import META_TITLE, ALIGNMENT_DEFINITIONS

from pydantic_models.segmentation import TaskGraph, IndexVideoSegmentation, get_segmentation_schema
from pydantic_models.summarization import MetaSummarySchema, SubgoalSummarySchema

from pydantic_models.comparison import AlignmentsSchema, SummarizedAlignmentSchema, NotableInformationSchema, HookSchema, AlignmentClassificationSchema, AlignmentHooksSchema

from helpers import get_response_pydantic, encode_image

