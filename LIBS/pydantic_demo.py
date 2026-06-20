from pydantic import BaseModel, Field
from typing import Optional

class StudentInfo(BaseModel):
    name: str = Field(..., description="The student's full name")
    age: Optional[int] = Field(None, description="The student's age")
    email: str = Field(..., description="The student's email address", pattern=r'^\S+@\S+\.\S+$')
    cgpa: Optional[float] = Field(None, description="The student's GPA", gt=0, lt=10.0)
    enrolled: bool = Field(..., description="Indicates if the student is currently enrolled")

stdt = StudentInfo(
    name="John Doe",
    age=20,
    email="john.doe@example.com",
    cgpa=3.5,
    enrolled=True
)

print(stdt.model_dump)
